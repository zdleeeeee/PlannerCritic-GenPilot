#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.absolute()
GENPILOT_DIR = PROJECT_ROOT / "genpilot"

load_dotenv(PROJECT_ROOT / ".env")

INPUT_FOLDER = PROJECT_ROOT / os.getenv("INPUT_FOLDER", "data")
BASE_OUTPUT_FOLDER = PROJECT_ROOT / os.getenv("OUTPUT_FOLDER", "results") / "baseline"
COMPARE_DIR = BASE_OUTPUT_FOLDER / "compare"
CASE_ID_FILE = PROJECT_ROOT / "case_ids.txt"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BATCH_COUNT = int(os.getenv("BATCH_COUNT", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
ERROR_ANALYSIS_WORKERS = os.getenv("ERROR_ANALYSIS_WORKERS", str(BATCH_SIZE))
DEFAULT_NUM_ITERATIONS = os.getenv("NUM_ITERATIONS", "4")
DEFAULT_L1_SCORE_THRESHOLD = os.getenv("L1_SCORE_THRESHOLD", "13.0")
DEFAULT_MAX_L1_RETRIES = os.getenv("MAX_L1_RETRIES", "2")
DEFAULT_L3_STAGNATION_WINDOW = os.getenv("L3_STAGNATION_WINDOW", "2")
DEFAULT_L3_SCORE_DELTA = os.getenv("L3_SCORE_DELTA", "1.0")


def run_command(cmd, cwd=GENPILOT_DIR):
    print("\n🚀", " ".join(str(part) for part in cmd))
    env = os.environ.copy()
    pythonpath_parts = [str(cwd)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def write_case_ids(batch_size):
    CASE_ID_FILE.write_text("".join(f"{idx}\n" for idx in range(batch_size)), encoding="utf-8")


def prepare_batch_prompts(batch_index):
    prompts_299 = INPUT_FOLDER / "original_prompts_299.txt"
    prompts = INPUT_FOLDER / "original_prompts.txt"
    start = batch_index * BATCH_SIZE
    end = start + BATCH_SIZE
    lines = prompts_299.read_text(encoding="utf-8").splitlines()[start:end]
    if len(lines) != BATCH_SIZE:
        raise ValueError(f"Batch {batch_index + 1} needs {BATCH_SIZE} prompts, got {len(lines)} from {prompts_299}")
    prompts.write_text("\n".join(lines) + "\n", encoding="utf-8")
    clear_ori_images()
    print(f"📝 Batch {batch_index + 1}/{BATCH_COUNT}: 写入 original_prompts_299.txt 第 {start}-{end - 1} 行到 {prompts}")


def clear_ori_images():
    image_dir = INPUT_FOLDER / "ori_img"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image_file in image_dir.glob("*.png"):
        image_file.unlink()


def run_error_analysis(analysis_dir):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--input_folder", str(INPUT_FOLDER),
        "--output_folder", str(analysis_dir),
        "--api_key", os.getenv("API_KEY", ""),
        "--url", os.getenv("BASE_URL", ""),
        "--api_model", os.getenv("API_MODEL", ""),
    ]
    worker_common = [*common, "--workers", ERROR_ANALYSIS_WORKERS]
    run_command([sys.executable, "error_analysis/decom_prompt.py", *common])
    run_command([sys.executable, "utils_all/reform_decomposed_prompt.py", "--input_file", str(analysis_dir / "decomposed_prompt.jsonl")])
    run_command([
        sys.executable, "error_analysis/generate_image.py",
        "--cuda", os.getenv("CUDA_DEVICE", "cuda:0"),
        "--input_folder", str(INPUT_FOLDER),
        "--model_name", os.getenv("MODEL_NAME", "flux"),
        "--model_path", os.getenv("MODEL_PATH", ""),
        "--api_key", os.getenv("API_KEY", ""),
        "--url", os.getenv("BASE_URL", ""),
        "--api_model", os.getenv("API_MODEL", ""),
    ])
    run_command([sys.executable, "error_analysis/caption.py", *worker_common])
    run_command([sys.executable, "error_analysis/check_captions.py", *worker_common])
    run_command([sys.executable, "utils_all/reform_jsonl.py", "--input_file", str(analysis_dir / "check_captions.jsonl")])
    run_command([sys.executable, "error_analysis/question.py", *worker_common])
    run_command([sys.executable, "error_analysis/qa.py", *worker_common])
    run_command([sys.executable, "error_analysis/error_integration.py", *worker_common])
    run_command([sys.executable, "utils_all/reform_jsonl.py", "--input_file", str(analysis_dir / "errors.jsonl")])
    run_command([sys.executable, "error_analysis/error_mapping.py", *worker_common])
    run_command([sys.executable, "utils_all/reform_jsonl.py", "--input_file", str(analysis_dir / "find_error.jsonl")])


def run_ttp_mode(mode, batch_dir, analysis_dir):
    mode_dir = batch_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "ttpo.py",
        "--case_id", str(CASE_ID_FILE),
        "--cuda", os.getenv("CUDA_DEVICE", "cuda:0"),
        "--input_folder", str(INPUT_FOLDER),
        "--output_folder", str(mode_dir),
        "--analysis_folder", str(analysis_dir),
        "--critic_mode", mode,
        "--model_name", os.getenv("MODEL_NAME", "flux"),
        "--model_path", os.getenv("MODEL_PATH", ""),
        "--api_key", os.getenv("API_KEY", ""),
        "--url", os.getenv("BASE_URL", ""),
        "--api_model", os.getenv("API_MODEL", ""),
        "--num_iterations", DEFAULT_NUM_ITERATIONS,
        "--num_candidates", os.getenv("NUM_CANDIDATES", "4"),
        "--num_clusters", os.getenv("NUM_CLUSTERS", "1"),
        "--num_gen_img", os.getenv("NUM_GEN_IMG", "1"),
        "--case_workers", os.getenv("CASE_WORKERS", str(BATCH_SIZE)),
        "--sentence_workers", os.getenv("SENTENCE_WORKERS", "2"),
        "--l1_score_threshold", DEFAULT_L1_SCORE_THRESHOLD,
        "--max_l1_retries", DEFAULT_MAX_L1_RETRIES,
        "--l3_stagnation_window", DEFAULT_L3_STAGNATION_WINDOW,
        "--l3_score_delta", DEFAULT_L3_SCORE_DELTA,
    ]
    run_command(cmd)
    return latest_result_dir(mode_dir)


def latest_result_dir(mode_dir):
    result_root = mode_dir / "result"
    candidates = [path for path in result_root.iterdir() if path.is_dir()]
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_score_line(line):
    text = line.strip()
    if text.startswith("[np.float64") and "{" not in text:
        return [float(value) for value in re.findall(r"np\.float64\(([-+]?\d+(?:\.\d+)?)\)", text)]
    if re.fullmatch(r"\[\s*[-+]?\d+(?:\.\d+)?(?:\s*,\s*[-+]?\d+(?:\.\d+)?)*\s*\]", text):
        return [float(value) for value in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    return []


def collect_metrics(result_dir):
    logs_dir = result_dir / "logs"
    score_values = []
    l1_count = 0
    l3_count = 0
    aggregator_count = 0
    for log_file in logs_dir.glob("*.txt"):
        text = log_file.read_text(encoding="utf-8", errors="ignore")
        l1_count += text.count("[L1] Trigger")
        l3_count += text.count("[L3] Trigger")
        aggregator_count += text.count("[ErrorAggregator]")
        for line in text.splitlines():
            numbers = parse_score_line(line)
            if numbers:
                score_values.append(max(numbers))
    prompt_file = result_dir / "prompt_final.txt"
    prompts = prompt_file.read_text(encoding="utf-8").splitlines() if prompt_file.exists() else []
    return {
        "result_dir": str(result_dir),
        "num_final_prompts": len([prompt for prompt in prompts if prompt.strip()]),
        "max_candidate_score": max(score_values) if score_values else None,
        "avg_round_best_score": sum(score_values) / len(score_values) if score_values else None,
        "num_scored_rounds": len(score_values),
        "l1_triggers": l1_count,
        "l3_triggers": l3_count,
        "error_aggregator_logs": aggregator_count,
    }


def write_batch_report(batch_dir, batch_index, baseline_dir, critic_dir):
    baseline_metrics = collect_metrics(baseline_dir)
    critic_metrics = collect_metrics(critic_dir)
    start = batch_index * BATCH_SIZE
    cases = list(range(start, start + BATCH_SIZE))
    report = {
        "timestamp": TIMESTAMP,
        "batch": batch_index + 1,
        "source_prompt_indices": cases,
        "local_case_ids": list(range(BATCH_SIZE)),
        "analysis_dir": str(batch_dir / "analysis"),
        "baseline": baseline_metrics,
        "critic": critic_metrics,
        "improvement": {
            "max_candidate_score_delta": _delta(critic_metrics["max_candidate_score"], baseline_metrics["max_candidate_score"]),
            "avg_round_best_score_delta": _delta(critic_metrics["avg_round_best_score"], baseline_metrics["avg_round_best_score"]),
        },
    }
    (batch_dir / "compare_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (batch_dir / "compare_summary.md").write_text(_report_markdown(report, baseline_dir, critic_dir), encoding="utf-8")
    return report


def write_overall_report(batch_reports):
    baseline_scores = [report["baseline"]["avg_round_best_score"] for report in batch_reports if report["baseline"]["avg_round_best_score"] is not None]
    critic_scores = [report["critic"]["avg_round_best_score"] for report in batch_reports if report["critic"]["avg_round_best_score"] is not None]
    overall = {
        "timestamp": TIMESTAMP,
        "batch_count": BATCH_COUNT,
        "batch_size": BATCH_SIZE,
        "total_prompts": BATCH_COUNT * BATCH_SIZE,
        "baseline_avg_round_best_score": _mean(baseline_scores),
        "critic_avg_round_best_score": _mean(critic_scores),
        "avg_round_best_score_delta": _delta(_mean(critic_scores), _mean(baseline_scores)),
        "batches": batch_reports,
    }
    (COMPARE_DIR / "compare_summary_all.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Baseline vs CriticPilot-Lite Overall Compare",
        "",
        f"- Timestamp: {TIMESTAMP}",
        f"- Batches: {BATCH_COUNT}",
        f"- Batch size: {BATCH_SIZE}",
        f"- Total prompts: {BATCH_COUNT * BATCH_SIZE}",
        "",
        "| Batch | Source prompt indices | Baseline avg | Critic avg | Delta | L1 | L3 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in batch_reports:
        lines.append(
            f"| {report['batch']} | {report['source_prompt_indices']} | "
            f"{_fmt(report['baseline']['avg_round_best_score'])} | "
            f"{_fmt(report['critic']['avg_round_best_score'])} | "
            f"{_fmt(report['improvement']['avg_round_best_score_delta'])} | "
            f"{report['critic']['l1_triggers']} | {report['critic']['l3_triggers']} |"
        )
    lines.extend([
        "",
        f"Overall baseline avg: {_fmt(overall['baseline_avg_round_best_score'])}",
        f"Overall critic avg: {_fmt(overall['critic_avg_round_best_score'])}",
        f"Overall delta: {_fmt(overall['avg_round_best_score_delta'])}",
    ])
    (COMPARE_DIR / "compare_summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report_markdown(report, baseline_dir, critic_dir):
    baseline_metrics = report["baseline"]
    critic_metrics = report["critic"]
    lines = [
        "# Baseline vs CriticPilot-Lite Compare",
        "",
        f"- Timestamp: {TIMESTAMP}",
        f"- Batch: {report['batch']}",
        f"- Source prompt indices: {report['source_prompt_indices']}",
        f"- Analysis dir: `{report['analysis_dir']}`",
        "",
        "| Metric | Baseline | Critic | Delta |",
        "| --- | ---: | ---: | ---: |",
        _metric_row("Max candidate score", baseline_metrics["max_candidate_score"], critic_metrics["max_candidate_score"]),
        _metric_row("Avg round-best score", baseline_metrics["avg_round_best_score"], critic_metrics["avg_round_best_score"]),
        _metric_row("Scored rounds", baseline_metrics["num_scored_rounds"], critic_metrics["num_scored_rounds"]),
        _metric_row("Final prompts", baseline_metrics["num_final_prompts"], critic_metrics["num_final_prompts"]),
        _metric_row("L1 triggers", baseline_metrics["l1_triggers"], critic_metrics["l1_triggers"]),
        _metric_row("L3 triggers", baseline_metrics["l3_triggers"], critic_metrics["l3_triggers"]),
        "",
        f"- Baseline result: `{baseline_dir}`",
        f"- Critic result: `{critic_dir}`",
    ]
    return "\n".join(lines) + "\n"


def _delta(new, old):
    if new is None or old is None:
        return None
    return new - old


def _mean(values):
    return sum(values) / len(values) if values else None


def _fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _metric_row(name, baseline, critic):
    return f"| {name} | {_fmt(baseline)} | {_fmt(critic)} | {_fmt(_delta(critic, baseline))} |"


def main():
    BASE_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    write_case_ids(BATCH_SIZE)
    print(f"📁 项目根目录: {PROJECT_ROOT}")
    print(f"📂 Compare 输出目录: {COMPARE_DIR}")
    print(f"🧪 运行设置: {BATCH_COUNT} 轮 × 每轮 {BATCH_SIZE} 个 prompts")
    print(f"⚙️  Batch 内并发: error_analysis_workers={ERROR_ANALYSIS_WORKERS}, case_workers={os.getenv('CASE_WORKERS', str(BATCH_SIZE))}, sentence_workers={os.getenv('SENTENCE_WORKERS', '2')}")
    print(f"🎯 Critic 触发参数: num_iterations={DEFAULT_NUM_ITERATIONS}, l1_score_threshold={DEFAULT_L1_SCORE_THRESHOLD}, l3_stagnation_window={DEFAULT_L3_STAGNATION_WINDOW}, l3_score_delta={DEFAULT_L3_SCORE_DELTA}")
    batch_reports = []
    try:
        for batch_index in range(BATCH_COUNT):
            batch_dir = COMPARE_DIR / f"batch_{batch_index + 1:02d}"
            analysis_dir = batch_dir / "analysis"
            batch_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n================ Batch {batch_index + 1}/{BATCH_COUNT} ================")
            prepare_batch_prompts(batch_index)
            run_error_analysis(analysis_dir)
            baseline_dir = run_ttp_mode("baseline", batch_dir, analysis_dir)
            critic_dir = run_ttp_mode("critic", batch_dir, analysis_dir)
            batch_reports.append(write_batch_report(batch_dir, batch_index, baseline_dir, critic_dir))
        write_overall_report(batch_reports)
        print(f"\n✅ {BATCH_COUNT} 轮 baseline/critic 跑完，总比较结果在: {COMPARE_DIR}")
    finally:
        if CASE_ID_FILE.exists():
            CASE_ID_FILE.unlink()


if __name__ == "__main__":
    main()
