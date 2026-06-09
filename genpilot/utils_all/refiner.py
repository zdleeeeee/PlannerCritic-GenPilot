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
from utils_all.error_taxonomy import build_strategy_block, classify_error

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def modify_sentence(client,error,sentence,history,template,candidates = []):
    categories = classify_error(error, sentence)
    strategy_block = build_strategy_block(categories)
    user_textprompt=f"""Error categories: {', '.join(categories)}
Category-specific modification strategy:
{strategy_block}

Input: \n Errors: \n {error} \n Sentence: \n {sentence} \n """
    textprompt= f"""{' '.join(template)} \n Revision history: {history} \n {user_textprompt}"""
    gen_text = client.request_gpt(textprompt)
    if gen_text is None:
        return "NONE"
    return gen_text.strip()

def merge_modified(client,prompt,sentence,template):
    user_textprompt=f"Input: \n Prompt: \n {prompt} \n Modified sentence: \n {sentence} \n "
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    prompt_after_merge_modified = client.request_gpt(textprompt)
    if prompt_after_merge_modified is None:
        return prompt
    return prompt_after_merge_modified.strip()


def llm_optimize(client,i,prompt_origin,image_data,history,j,template_his,template_merge):
    # print(type(i))
    sentence = image_data[i]['sentence'][str(j+1)]
    error = image_data[i]['errors'][str(j+1)]
    modified_sentence = modify_sentence(client,error,sentence,history,template_his)
    print(f"{i}-{modified_sentence}")
    if modified_sentence and modified_sentence.upper() != "NONE":
        modified_prompt = merge_modified(client,prompt_origin,modified_sentence,template_merge)
        return modified_prompt,modified_sentence
    else:
        return "NO ERROR","NO ERROR"

def generate_candidate_prompts(client,i,prompt,image_data,history,j,template_his,template_merge, num_candidates):
    prompts = []
    sentences = []
    seen_prompts = set()
    for _ in range(num_candidates):
        # print(g)
        modified_prompt,modified_sentence = llm_optimize(client,i,prompt,image_data,history,j,template_his,template_merge)
        if modified_prompt == "NO ERROR" and modified_sentence == "NO ERROR":
            return "NO ERROR","NO ERROR"
        if not modified_prompt or modified_prompt in seen_prompts:
            continue
        seen_prompts.add(modified_prompt)
        prompts.append(modified_prompt)
        sentences.append(modified_sentence)
    if not prompts:
        return "NO ERROR","NO ERROR"
    print(prompts)
    # while len(prompts) < num_clusters:
    #     modified_prompt, modified_sentence = llm_optimize(i, prompt, image_data, history, j, template_his, template_merge)
    #     if modified_prompt == "NO ERROR" and modified_sentence == "NO ERROR":
    #         return "NO ERROR", "NO ERROR"
    #     prompts.append(modified_prompt)
    #     sentences.append(modified_sentence)
    return prompts,sentences
    # return [llm_optimize(prompt,image_data,history,j,template_his,template_merge) for _ in range(num_candidates)]
