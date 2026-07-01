from urllib.parse import quote_plus
from dotenv import load_dotenv
import os

load_dotenv()

# ── PostgreSQL ────────────────────────────────────────────────────────────────
PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = os.getenv("PG_PORT",     "5432")
PG_DATABASE = os.getenv("PG_DATABASE", "olist_bi")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

PG_URL = f"postgresql+psycopg2://{PG_USER}:{quote_plus(PG_PASSWORD)}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MLFLOW_DIR  = os.path.join(BASE_DIR, "mlflow_tracking")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DIR}/mlflow.db"
MLFLOW_EXPERIMENT   = "olist_bi_pipeline"
