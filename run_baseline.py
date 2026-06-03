#!/usr/bin/env python3
"""
run_baseline.py - 简洁的基线测试脚本
运行 GenPilot 的 TTPO 并记录结果
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.absolute()
GENPILOT_DIR = PROJECT_ROOT / "genpilot"

sys.path.insert(0, str(GENPILOT_DIR))

load_dotenv(PROJECT_ROOT / ".env")

# ========== 配置 ==========
INPUT_FOLDER = PROJECT_ROOT / os.getenv("INPUT_FOLDER", "data")
OUTPUT_FOLDER = PROJECT_ROOT / os.getenv("OUTPUT_FOLDER", "results")
CASE_IDS = os.getenv("CASE_IDS", "0,1,2,3,4")
CASE_ID_FILE = PROJECT_ROOT / "case_ids.txt"  # 临时文件

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_FOLDER / f"baseline"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ========== 准备 case_id 文件 ==========
with open(CASE_ID_FILE, "w") as f:
    for idx in CASE_IDS.split(","):
        f.write(f"{idx.strip()}\n")

print(f"📁 项目根目录: {PROJECT_ROOT}")
print(f"📂 输入目录: {INPUT_FOLDER}")
print(f"📂 输出目录: {RUN_DIR}")
print(f"📝 测试用例 ID: {CASE_IDS}")
print("=" * 50)

# ========== 构建命令 ==========
cmd = [
    sys.executable,  # 当前 Python 解释器
    str("ttpo.py"),
    "--case_id", str(CASE_ID_FILE),
    "--cuda", os.getenv("CUDA_DEVICE", "cuda:0"),
    "--input_folder", str(INPUT_FOLDER),
    "--output_folder", str(RUN_DIR),
    "--model_name", os.getenv("MODEL_NAME", "flux"),
    "--model_path", os.getenv("MODEL_PATH", ""),
    "--api_key", os.getenv("API_KEY", ""),
    "--url", os.getenv("BASE_URL", ""),
    "--api_model", os.getenv("API_MODEL", ""),
]

print("\n🚀 运行命令:")
print(" ".join(cmd))
print("\n" + "=" * 50)

# ========== 运行 ==========
try:
    result = subprocess.run(
        cmd,
        cwd=GENPILOT_DIR,
        capture_output=False,
        text=True,
        check=True)
    print("\n✅ 基线测试完成！")
    print(f"📁 结果保存在: {RUN_DIR}")
    
    info = {
        "timestamp": TIMESTAMP,
        "case_ids": CASE_IDS.split(","),
        "model_name": os.getenv("MODEL_NAME"),
        "output_dir": str(RUN_DIR),
    }
    with open(RUN_DIR / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
        
except subprocess.CalledProcessError as e:
    print(f"\n❌ 运行失败: {e}")
    sys.exit(1)

# ========== 清理临时文件 ==========
if CASE_ID_FILE.exists():
    CASE_ID_FILE.unlink()