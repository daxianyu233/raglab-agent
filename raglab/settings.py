"""RAGLab 项目的统一路径配置。"""

from pathlib import Path


# rag-lab 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据
DATA_DIR = PROJECT_ROOT / "data"
PDF_CORPUS_DIR = DATA_DIR / "corpus" / "pdf"
EVAL_DATASET_PATH = DATA_DIR / "eval" / "eval_dataset.json"
CORPUS_MANIFEST_PATH = DATA_DIR / "metadata" / "corpus_manifest.json"

# 实验配置
CONFIG_DIR = PROJECT_ROOT / "config"
BASELINE_CONFIG_PATH = CONFIG_DIR / "baseline.yaml"

# 可重新生成的数据
STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"

# 实验输出
REPORTS_DIR = PROJECT_ROOT / "reports"