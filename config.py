from pathlib import Path

# ======================
# 项目根目录（唯一标准）
# ======================
ROOT_DIR = Path(__file__).resolve().parent

# ======================
# 各模块统一路径
# ======================
EDGE_DIR = ROOT_DIR / "edge"
CLOUD_DIR = ROOT_DIR / "cloud"
DASHBOARD_DIR = ROOT_DIR / "dashboard"

LATEST_IMAGE = DASHBOARD_DIR / "latest.jpg"
LATEST_TMP = DASHBOARD_DIR / "latest.tmp.jpg"
STATUS_FILE = ROOT_DIR / "status.json"

KNOWLEDGE_DIR = ROOT_DIR / "knowledge_base"
CHROMA_DIR = ROOT_DIR / "chroma_db"