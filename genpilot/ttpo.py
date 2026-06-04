import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from PIL import Image
import base64
from openai import OpenAI
# from gpt_proxy import OpenAIApiProxy
import os
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from numpy import mean
import random
from datetime import datetime
import concurrent.futures
from tqdm import tqdm
import threading
import torch
from PIL import Image
import random
import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
from utils_all.loading import logger_print_txt,load_json,load_txt,load_jsonl,get_image_paths
from utils_all.api import APIClient
from utils_all.scorer import scoring_function,update_history
from utils_all.refiner import generate_candidate_prompts,merge_modified
from utils_all.cluster import cluster_prompts,update_probabilities
from utils_all.error_taxonomy import aggregate_errors, count_error_items
# 用于同步 a、b、c 的锁
lock_a = threading.Lock()
import importlib
import functools
def log_time_with_logger():
    """
    装饰器：记录函数运行时间。
    使用方法：
        @log_time_with_logger()
        def your_function(...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            duration = end - start
            print(f"🟢 [Time] {func.__name__} 耗时: {duration:.3f} 秒")
            return result
        return wrapper
    return decorator

def match_image_with_text(image_folder, caption_file, prompt_file):
    bugs = load_jsonl(caption_file)
    prompt = load_jsonl(prompt_file)
    images = get_image_paths(image_folder)

    # 验证匹配并打印结果
    if len(images) == len(bugs) == len(prompt):
        print("Images and JSONL files are correctly matched by length.")
    else:
        print(f"Mismatch: {len(images)} images, {len(bugs)} entries in JSONL caption, {len(prompt)} entries in JSONL prompt.")

    # 示例：关联图像和JSONL内容
    image_data = [
        {
            "sentence": prompt[i],
            "errors": bugs[i]
        }
        for i in range(len(images))
    ]
    return image_data

def load_model_module(name):
    if name == "flux":
        module_name = "utils_all.image_flux"
    elif name in ["sd1", "sd2", "sd3"]:
        module_name = f"utils_all.image_{name}"
    else:
        raise ValueError(f"Unknown model name: {name}")

    return importlib.import_module(module_name)

def generate_image(path,prompt,k):
    with lock_a:
        # pipline = get_pipe_slow("cuda:1")
        imgs = model_module.t2i_slow(pipeline, prompt, k,output_dir=path)
    # torch.cuda.empty_cache()
    return [path+f"/{k}.png"]

@log_time_with_logger()
def gen_batch_image(num,path,prompt):
    # pipeline = get_pipe_slow()
    img_list = []
    with lock_a:
        prompt_li = [prompt for _ in range(num)]    
        # pipline = get_pipe_slow("cuda:1")
        imgs = model_module.t2i_slow_batch(pipeline, prompt_li, output_dir=path)
        for i, image in enumerate(imgs):
            image.save(path+f"/{i}.png")
            img_list.append(path+f"/{i}.png")
        # torch.cuda.empty_cache()
        return img_list


def load_template():
    with open('prompts/refine.txt', 'r', encoding='utf-8') as f:
        template=f.readlines()
    with open('prompts/refine_with_history.txt', 'r', encoding='utf-8') as f:
        template_his=f.readlines()
    with open('prompts/merge.txt', 'r', encoding='utf-8') as f:
        template_merge=f.readlines()
    with open('prompts/qa_4_check.txt', 'r', encoding='utf-8') as f:
        template_ques=f.readlines()
    with open('prompts/rate_feedback.txt', 'r', encoding='utf-8') as f:
        template_rate=f.readlines()
    with open('prompts/feed_back_reasons.txt', 'r', encoding='utf-8') as f:
        template_reason=f.readlines()
    with open('prompts/refine_with_history_with_multi.txt', 'r', encoding='utf-8') as f:
        template_refine_multi=f.readlines()
    return template,template_his,template_merge,template_ques,template_rate,template_reason,template_refine_multi
def datetime_now():
# 获取当前时间
    current_time = datetime.now()

    # 格式化为“年月日时分秒”格式
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")

    return formatted_time


def _dedupe_candidate_pairs(candidate_prompts, modified_sentences):
    deduped_prompts = []
    deduped_sentences = []
    seen_prompts = set()
    for prompt, sentence in zip(candidate_prompts, modified_sentences):
        if not prompt or prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        deduped_prompts.append(prompt)
        deduped_sentences.append(sentence)
    return deduped_prompts, deduped_sentences


def aggregate_case_errors(case_data, log_file, max_errors_per_fragment=5):
    for key, errors in case_data["errors"].items():
        sentence = case_data["sentence"].get(key, "") if isinstance(case_data["sentence"], dict) else ""
        original_count = count_error_items(errors)
        aggregated = aggregate_errors(errors, sentence, max_errors_per_fragment)
        case_data["errors"][key] = aggregated
        logger_print_txt(log_file, f"[ErrorAggregator] fragment={key}, errors={original_count}->{len(aggregated)}")


@log_time_with_logger()
def process_single_j(j, i, image_data, question, initial_prompt, root_folder, log_file, prompt_origin, template_his, template_merge, num_iterations, num_candidates, num_gen_img, use_history, prior_probabilities, enable_l1=True, enable_l3=True, l1_score_threshold=13.0, max_l1_retries=2, l3_stagnation_window=2, l3_score_delta=1.0):
    question_list = question[i][str(j + 1)]
    history = []
    skip_j = False
    modify_history = []
    init_tmp = ""
    modify_sentence_tmp = None
    rate_sentence_k = []
    stagnation_counter = 0
    last_best_score = None

    def evaluate_candidates(candidate_prompts, modified_sentences, iteration_index, retry_index=0):
        candidate_prompts, modified_sentences = _dedupe_candidate_pairs(candidate_prompts, modified_sentences)
        logger_print_txt(log_file, f"candidate_prompts in {i}-{j}-{iteration_index}-retry{retry_index}-{candidate_prompts}")
        logger_print_txt(log_file, modified_sentences)
        scores_local = []
        rate_sentence_local = []
        for z, candidate_prompt in enumerate(candidate_prompts):
            image_folder_k = f"{root_folder}/imgs/{i}/{j}/{iteration_index}/{retry_index}_{z}"
            os.makedirs(image_folder_k, exist_ok=True)
            image = gen_batch_image(num_gen_img, image_folder_k, candidate_prompt)
            logger_print_txt(log_file, f"[Info] generate_image for candidate_prompts")
            score, rate_result = scoring_function(client,image_data,i, j, candidate_prompt, image, prompt_origin, question_list,template_ques,template_rate,template_reason)
            logger_print_txt(log_file, f"[Info] scoring_function for candidate_prompts")
            rate_sentence_local.append(rate_result)
            scores_local.append(score)
        return candidate_prompts, modified_sentences, scores_local, rate_sentence_local

    for k in range(num_iterations):
        logger_print_txt(log_file, f"================{i}-{j}-{k}==============")
        logger_print_txt(log_file, initial_prompt)

        candidate_prompts, modified_sentence = generate_candidate_prompts(client,i, initial_prompt, image_data, history, j, template_his, template_merge, num_candidates)
        if candidate_prompts == "NO ERROR" and modified_sentence == "NO ERROR":
            skip_j = True
            break
        logger_print_txt(log_file, f"[Info] generate_candidate_prompts.")
        candidate_prompts, modified_sentence, scores, rate_sentence_k = evaluate_candidates(candidate_prompts, modified_sentence, k)

        l1_retry_count = 0
        if enable_l1 and scores:
            logger_print_txt(log_file, f"[L1Check] max_score={max(scores)}, threshold={l1_score_threshold}, will_retry={max(scores) < l1_score_threshold}")
        while enable_l1 and scores and max(scores) < l1_score_threshold and l1_retry_count < max_l1_retries:
            l1_retry_count += 1
            logger_print_txt(log_file, f"[L1] Trigger local retry {l1_retry_count}/{max_l1_retries}: max_score={max(scores)}, threshold={l1_score_threshold}")
            candidate_prompts_retry, modified_sentence_retry = generate_candidate_prompts(client,i, initial_prompt, image_data, history, j, template_his, template_merge, num_candidates * 2)
            if candidate_prompts_retry == "NO ERROR" and modified_sentence_retry == "NO ERROR":
                logger_print_txt(log_file, f"[L1] Retry returned NO ERROR; keeping current candidates.")
                break
            candidate_prompts, modified_sentence, scores, rate_sentence_k = evaluate_candidates(candidate_prompts_retry, modified_sentence_retry, k, l1_retry_count)
            if scores:
                logger_print_txt(log_file, f"[L1] Retry complete: new_max_score={max(scores)}")

        if not scores:
            skip_j = True
            break

        modify_history.append(modified_sentence)
        current_best_score = max(scores)
        if enable_l3:
            if last_best_score is not None:
                score_change = current_best_score - last_best_score
                if abs(score_change) < l3_score_delta:
                    stagnation_counter += 1
                else:
                    stagnation_counter = 0
                logger_print_txt(log_file, f"[L3Check] current_best={current_best_score}, last_best={last_best_score}, score_change={score_change}, stagnation={stagnation_counter}/{l3_stagnation_window}")
            else:
                logger_print_txt(log_file, f"[L3Check] current_best={current_best_score}, last_best=None, stagnation=0/{l3_stagnation_window}")
            last_best_score = current_best_score

            if stagnation_counter >= l3_stagnation_window:
                logger_print_txt(log_file, f"[L3] Trigger global reset: stagnation={stagnation_counter}, max_score={current_best_score}")
                history = []
                modify_history = []
                stagnation_counter = 0
                last_best_score = None

        logger_print_txt(log_file, scores)
        print(f"{i}-{candidate_prompts}")
        effective_num_clusters = max(1, min(num_clusters, len(candidate_prompts)))
        labels = cluster_prompts(candidate_prompts, effective_num_clusters)

        if len(set(labels)) == 1:
            best_cluster_prompts = candidate_prompts
            logger_print_txt(log_file, f"[Info] All prompts assigned to one cluster.")
            best_prompt_index = int(np.argmax(scores))
            score_4_judge = scores
            modify_sentence_tmp = modified_sentence[best_prompt_index] if best_prompt_index < len(modified_sentence) else "None"
            print(f"{modify_sentence_tmp}-{modified_sentence}-{best_prompt_index}")
        else:
            effective_prior_probabilities = prior_probabilities[:effective_num_clusters]
            if len(effective_prior_probabilities) != effective_num_clusters:
                effective_prior_probabilities = [1 / effective_num_clusters] * effective_num_clusters
            posterior_probabilities = update_probabilities(scores, labels, effective_prior_probabilities)
            best_cluster_index = np.argmax(posterior_probabilities)
            best_cluster_prompts = [candidate_prompts[q] for q, label in enumerate(labels) if label == best_cluster_index]

            image_folder_k_cluster = f"{root_folder}/cluster/{i}/{j}/{k}"
            os.makedirs(image_folder_k_cluster, exist_ok=True)

            scores_here = []
            for z in range(len(best_cluster_prompts)):
                image = generate_image(image_folder_k_cluster, best_cluster_prompts[z], z)
                logger_print_txt(log_file, f"[Info] generate_image for best_cluster_prompts")
                score, rate_result = scoring_function(client,image_data,i, j, best_cluster_prompts[z], image, prompt_origin, question_list,template_ques,template_rate,template_reason)
                logger_print_txt(log_file, f"[Info] scoring_function for best_cluster_prompts")
                scores_here.append(score)

            while len(scores_here) == 0:
                for z in range(len(best_cluster_prompts)):
                    image = generate_image(image_folder_k_cluster, best_cluster_prompts[z], z)
                    score, rate_result = scoring_function(client,image_data,i, j, best_cluster_prompts[z], image, prompt_origin, question_list,template_ques,template_rate,template_reason)
                    scores_here.append(score)

            prior_probabilities = posterior_probabilities
            best_prompt_index = int(np.argmax(scores_here))
            score_4_judge = scores_here
            modify_sentence_tmp = "None"
        best_prompt = best_cluster_prompts[best_prompt_index]
        init_tmp = best_prompt
        initial_prompt = best_prompt
        logger_print_txt(log_file, f"best prompt in {i}-{j}-{k}")
        logger_print_txt(log_file, best_prompt)

        if score_4_judge[best_prompt_index] / 3 >= 4.7:
            break

        if use_history:
            if len(modified_sentence) != len(rate_sentence_k):
                logger_print_txt(log_file, f"len(modified_sentence) != len(rate_sentence_k)")
            for p in range(min(len(rate_sentence_k), len(modified_sentence))):
                history_avg = update_history(modified_sentence[p], rate_sentence_k[p])
                history.append(history_avg)
            logger_print_txt(log_file, history)

    initial_prompt = init_tmp
    if skip_j:
        return j, None, None, None

    return j, rate_sentence_k,initial_prompt, modify_sentence_tmp



def process_case(i, prompt_origin, prior_probabilities, enable_error_aggregator=True, enable_l1=True, enable_l3=True, l1_score_threshold=13.0, max_l1_retries=2, l3_stagnation_window=2, l3_score_delta=1.0):
    promptlist_all = [None] * len(image_data[i]["sentence"])
    rate_sentence_all = [None] * len(image_data[i]["sentence"])
    modify_history_all = [None] * len(image_data[i]["sentence"])

    log_file = log_file_all + f"/{i}.txt"
    if enable_error_aggregator:
        aggregate_case_errors(image_data[i], log_file)
    initial_prompt = prompt_origin

    with concurrent.futures.ThreadPoolExecutor(max_workers=sentence_workers) as executor:
        futures = {
            executor.submit(
                process_single_j, j, i, image_data, question, initial_prompt,
                root_folder, log_file, prompt_origin, template_his, template_merge,
                num_iterations, num_candidates, num_gen_img, use_history,
                prior_probabilities, enable_l1, enable_l3, l1_score_threshold,
                max_l1_retries, l3_stagnation_window, l3_score_delta
            ): j
            for j in range(len(image_data[i]["sentence"]))
        }
        for future in concurrent.futures.as_completed(futures):
        # for future in concurrent.futures.as_completed(futures):
            try:
                j, rate_sentence_j, prompt_j, history_j = future.result()
                rate_sentence_all[j] = rate_sentence_j
                promptlist_all[j] = prompt_j
                modify_history_all[j] = history_j
            except Exception as e:
                print(f"[Error] Case {i} - j={j} failed: {e}")
                traceback.print_exc()

    logger_print_txt(log_file, "================promptlist after==============")
    logger_print_txt(log_file, str(promptlist_all))

    # # ===== 清空 i 目录下的 .png 图片 =====
    # for root, dirs, files in os.walk(f"{root_folder}/imgs/{i}"):
    #     for file in files:
    #         if file.endswith('.png'):
    #             file_path = os.path.join(root, file)
    #             try:
    #                 os.remove(file_path)
    #                 print(f"已删除文件: {file_path}")
    #             except Exception as e:
    #                 print(f"删除文件 {file_path} 时出错: {e}")
    
    valid_prompts = [prompt for prompt in promptlist_all if prompt]
    prompt_after = valid_prompts[0] if valid_prompts else prompt_origin
    for q in range(len(promptlist_all)):
        if not promptlist_all[q] and not modify_history_all[q]:
            continue
        if modify_history_all[q] == "None":
            prompt_after = merge_modified(client,prompt_after,promptlist_all[q],template_merge)
        elif modify_history_all[q]:
            prompt_after = merge_modified(client,prompt_after,modify_history_all[q],template_merge)
    logger_print_txt(log_file,"================prompt after==============")
    logger_print_txt(log_file,str(prompt_after))
    return i, prompt_after


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running TTPO")
    parser.add_argument('--case_id', type=str, required=True, help='Path to the case_id txt file')    
    parser.add_argument('--cuda', type=str, required=True, help='cuda devices') 
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input file') 
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output file') 
    parser.add_argument('--model_name', type=str, required=True, help='Model name: flux, sd1, sd2, sd3')
    parser.add_argument('--model_path', type=str, required=True, help='Model path: flux, sd1, sd2, sd3')
    parser.add_argument('--api_key', type=str, required=True, help='Api_key for API request')
    parser.add_argument('--url', type=str, required=True, help='Base_url for API request')
    parser.add_argument('--api_model', type=str, required=True, help='Model for API request')
    parser.add_argument('--num_iterations', type=int, default=2, help='Max refinement rounds per prompt sentence')
    parser.add_argument('--num_candidates', type=int, default=4, help='Candidate prompts generated per round')
    parser.add_argument('--num_clusters', type=int, default=1, help='Clusters used for candidate selection')
    parser.add_argument('--num_gen_img', type=int, default=1, help='Images generated per candidate for scoring')
    parser.add_argument('--case_workers', type=int, default=1, help='Parallel cases to process')
    parser.add_argument('--sentence_workers', type=int, default=2, help='Parallel prompt sentences per case')
    parser.add_argument('--critic_mode', choices=['baseline', 'critic'], default='critic', help='Run original baseline controls or CriticPilot-Lite controls')
    parser.add_argument('--analysis_folder', type=str, default=None, help='Folder containing errors_reformed.jsonl, find_error.jsonl, and questions.jsonl')
    parser.add_argument('--l1_score_threshold', type=float, default=13.0, help='Critic L1 retry threshold on total 3-15 score')
    parser.add_argument('--max_l1_retries', type=int, default=2, help='Maximum L1 retries per iteration')
    parser.add_argument('--l3_stagnation_window', type=int, default=2, help='Consecutive stagnant rounds before L3 reset')
    parser.add_argument('--l3_score_delta', type=float, default=1.0, help='Score delta below which a round is stagnant')
    # 解析命令行参数
    args = parser.parse_args()
    model_module = load_model_module(args.model_name)
    input_folder = args.input_folder
    output_folder = args.output_folder
    analysis_folder = args.analysis_folder or args.output_folder
    api_key = args.api_key
    url = args.url
    api_model = args.api_model
    model_path = args.model_path

    pipeline = model_module.get_pipe_slow(model_path,args.cuda)
    data1 = load_txt(args.case_id)
    data_num = []
    for i in data1:
        data_num.append(int(i))
    client = APIClient(api_key ,url, api_model)
    # device = "cuda:4"
    prompt_file_name = f"result_case_{data_num[0]}_{data_num[-1]}_avg_{datetime_now()}"
    template,template_his,template_merge,template_ques,template_rate,template_reason,template_refine_multi = load_template()
    image_folder = f"{input_folder}/ori_img"
    jsonl_caption = f"{analysis_folder}/errors_reformed.jsonl"
    jsonl_prompt = f"{analysis_folder}/find_error.jsonl"
    question_file = f"{analysis_folder}/questions.jsonl"
    txt_ori = load_txt(f"{input_folder}/original_prompts.txt")
    
    root_folder = f'{output_folder}/result/{prompt_file_name}'

    output_sentence_file = f'{root_folder}/refine_sentence.txt'
    output_prompt_file = f'{root_folder}/refine_prompt.txt'
    output_rate_file = f'{root_folder}/rate_sentence.txt'
    output_prompt_merged_file =f'{root_folder}/prompt_final.txt'
    # output_prompt_merged_file =f'{root_folder}/prompt_final.txt'
    output_prompt_merged_file_json = f'{root_folder}/prompt_final.json'
    log_file_all = f"{root_folder}/logs"
    log_file_after =  f"{root_folder}/logs_all.txt"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    if not os.path.exists(log_file_all):
        os.makedirs(log_file_all)


    image_data = match_image_with_text(image_folder, jsonl_caption, jsonl_prompt)
    question = load_jsonl(question_file)


    qa_check_flag = True
    generate_img = True
    rate_flag = True
    delta_revise = False
    num_gen_img = args.num_gen_img
    enable_critic = args.critic_mode == 'critic'
    enable_error_aggregator = enable_critic
    enable_l1 = enable_critic
    enable_l3 = enable_critic
    use_history = enable_critic
    num_4_select_prompt = 3
    results = []
    num_iterations = args.num_iterations
    num_candidates = args.num_candidates
    num_clusters = max(1, min(args.num_clusters, num_candidates))
    case_workers = max(1, args.case_workers)
    sentence_workers = max(1, args.sentence_workers)

    print(f"========================critic_mode:{args.critic_mode},analysis_folder:{analysis_folder},num_4_select_prompt:{num_4_select_prompt},num_iterations:{num_iterations},num_candidates:{num_candidates},num_clusters:{num_clusters},num_gen_img:{num_gen_img},case_workers:{case_workers},sentence_workers:{sentence_workers},l1_score_threshold:{args.l1_score_threshold},max_l1_retries:{args.max_l1_retries},l3_stagnation_window:{args.l3_stagnation_window},l3_score_delta:{args.l3_score_delta}============================")
    prior_probabilities = [1 / num_clusters] * num_clusters
    prompt_refined = {}
    results = []
    start_now = datetime.now()
    with concurrent.futures.ThreadPoolExecutor(max_workers=case_workers) as executor:
        futures = {executor.submit(process_case, i, txt_ori[i], prior_probabilities, enable_error_aggregator, enable_l1, enable_l3, args.l1_score_threshold, args.max_l1_retries, args.l3_stagnation_window, args.l3_score_delta): i for i in data_num}
        with tqdm(total=len(futures), desc="Processing Cases", unit="case") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    i, result = future.result()
                    results.append((i, result))
                except Exception as e:
                    print(f"Error processing case {futures[future]}: {e}")
                    traceback.print_exc()
                pbar.update(1)

    results.sort(key=lambda x: x[0])
    with open(output_prompt_merged_file, "w", encoding="utf-8") as file:
        for _, prompt in results:
            file.write(prompt + "\n")
    json_data = {str(i): prompt for i, prompt in results}
        # 写入 JSON 文件
    with open(output_prompt_merged_file_json, "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)
    logger_print_txt(log_file_after,f"log in {log_file_all}")
    end_now = datetime.now()
    print(f"start at {start_now.strftime('%Y-%m-%d %H:%M:%S')} and end at {end_now.strftime('%Y-%m-%d %H:%M:%S')}")

