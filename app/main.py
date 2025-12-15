from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
from joblib import load
import pandas as pd
from pathlib import Path
import numpy as np
import shap
import csv, json
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import APIRouter
from app.services.feature_engineering import make_datasets
from app.services.model_train import train as train_model
from app.services.model_train import train_claims as train_claims_model
from app.schemas import ClaimInput, DecisionOutput
from app.services.model_infer import load_model, explain
from starlette.responses import HTMLResponse
from fastapi.responses import FileResponse, HTMLResponse
from app.services.model_infer import load_model

app = FastAPI(title="Claims MVP", version="0.4.0")
admin = APIRouter(prefix="/admin", tags=["admin"])

@admin.post("/prep")
def prep_data():
    events, feats = make_datasets("data")
    return {"events": len(events), "features": len(feats)}

@admin.post("/train")
def train():
    pipe = train_model("data", "models")
    return {"status": "ok", "model": "models/model.pkl"}

@admin.post("/train_claims")
def train_claims():
    pipe = train_claims_model("data", "models")
    return {"status": "ok", "model": "models/model_claims.pkl"}

# ==== Schemas ====
class Features(BaseModel):
    claim_amount: Optional[float] = None
    pet_age_months: Optional[int] = None
    text: Optional[str] = None  # 暂未用于建模

class ScoreRequest(BaseModel):
    claim_id: Optional[str] = None
    features: Optional[Features] = None
    model_version: Optional[str] = None

class Prediction(BaseModel):
    label: str
    score: float

class ExplainRequest(BaseModel):
    claim_id: Optional[str] = None
    features: Optional[Features] = None
    model_version: str
    prediction: Prediction

class TopFeature(BaseModel):
    name: str
    importance: float
    contribution: float

class ScoreResponse(BaseModel):
    label: str
    score: float
    model_version: str

class ExplainResponse(BaseModel):
    top_features: List[TopFeature]
    method: str
    stability_score: Optional[float] = None
    plot_uri: Optional[str] = None

class OverrideRequest(BaseModel):
    claim_id: Optional[str] = None
    features: Optional[Features] = None
    new_label: str  # approve / reject / review
    note: Optional[str] = None
    actor: Optional[str] = "adjuster@demo"

class OverrideResponse(BaseModel):
    ok: bool
    decision_id: str

# ==== Load model at startup ====
# MODEL_PATH = Path("models/model.joblib")
# ART = load(MODEL_PATH)
# PIPE = ART["pipeline"]
# FEATURE_ORDER = ART["feature_order"]
# MODEL_VERSION = ART["model_version"]

# ==== Data & files ====
Path("data").mkdir(exist_ok=True)
DECISIONS_CSV = Path("data/decisions.csv")
AUDIT_CSV = Path("data/audit_log.csv")
PET_FEATURES_CSV = Path("data/processed/pet_features.csv")
AGE_CAP_MONTHS = 240.0

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_csv(file: Path, headers: list[str]) -> None:
    if not file.exists():
        with file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_csv(file: Path, headers: list[str], row: dict) -> None:
    ensure_csv(file, headers)
    with file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writerow({k: row.get(k, "") for k in headers})

def _map_prob_to_label(p_risky: float) -> str:
    if p_risky < 0.30:
        return "approve"
    elif p_risky > 0.70:
        return "reject"
    else:
        return "review"

def _load_model_or_http(model_key: str = "underwriting"):
    try:
        return load_model(model_key=model_key)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"Model artifacts not found for model_key='{model_key}'. Please train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model artifacts: {type(e).__name__}: {e}")

def _select_model_key(payload: ClaimInput, model_key: Optional[str]) -> str:
    if model_key:
        return model_key
    for k in ("amt_sum_12m", "amt_mean", "amt_max", "n_claims"):
        if getattr(payload, k, None) is not None:
            return "claims"
    return "underwriting"

def _apply_age_handling(payload: ClaimInput, X: pd.DataFrame) -> float:
    age_penalty = 0.0

    age_raw = getattr(payload, "pet_age_months", None)
    if age_raw is None or "pet_age_months" not in X.columns:
        return 0.0
    try:
        age_val = float(age_raw)
    except Exception:
        return 0.0

    if age_val < 1.0:
        age_val = 1.0
    if age_val > AGE_CAP_MONTHS:
        # Scheme A: stronger monotonic penalty for out-of-distribution ages
        over = max(0.0, age_val - AGE_CAP_MONTHS)
        tau = max(1.0, AGE_CAP_MONTHS / 4.0)
        age_penalty = min(0.80, (1.0 - float(np.exp(-over / tau))) * 0.80)
        age_val = AGE_CAP_MONTHS

    X.loc[X.index[0], "pet_age_months"] = age_val
    return float(age_penalty)

def _load_options() -> dict:
    default = {
        "species": ["Dog", "Cat"],
        "breeds_by_species": {"Dog": [], "Cat": []},
        "enroll_paths": ["Online", "Agent", "Phone", "Vet"],
        "age_cap_months": AGE_CAP_MONTHS,
    }
    if not PET_FEATURES_CSV.exists():
        return default
    try:
        df = pd.read_csv(PET_FEATURES_CSV)
    except Exception:
        return default

    species = default["species"]
    if "Species" in df.columns:
        s = sorted([x for x in df["Species"].dropna().astype(str).unique().tolist() if x])
        if s:
            species = s

    breeds_by_species: dict[str, list[str]] = {}
    if "Species" in df.columns and "Breed" in df.columns:
        tmp = df[["Species", "Breed"]].dropna()
        tmp["Species"] = tmp["Species"].astype(str)
        tmp["Breed"] = tmp["Breed"].astype(str)
        for sp, g in tmp.groupby("Species"):
            breeds = sorted([x for x in g["Breed"].unique().tolist() if x])
            breeds_by_species[str(sp)] = breeds

    enroll_paths = default["enroll_paths"]
    if "EnrollPath" in df.columns:
        ep = sorted([x for x in df["EnrollPath"].dropna().astype(str).unique().tolist() if x])
        if ep:
            enroll_paths = ep

    return {
        "species": species,
        "breeds_by_species": breeds_by_species,
        "enroll_paths": enroll_paths,
        "age_cap_months": AGE_CAP_MONTHS,
    }

def _nl_explain_en(topk: list[tuple[str, float]], payload) -> str:
    """
    Convert top feature contributions into short English reasons.
    Positive value = raises risk; negative = lowers risk.
    """
    msgs = []

    # Pull raw inputs (may be None)
    n = getattr(payload, "n_claims", None)
    s = getattr(payload, "amt_sum_12m", None)
    a = getattr(payload, "amt_mean", None)
    m = getattr(payload, "amt_max", None)

    for name, val in topk:
        base = name.split("__", 1)[-1]  # strip 'num__/cat__' prefix if present
        raises = (val > 0)

        if base.startswith("n_claims"):
            suffix = f" is {n}" if n is not None else ""
            msgs.append(f"Claim frequency in the last 12 months{suffix} "
                        f"{'increases' if raises else 'reduces'} risk.")
        elif base.startswith("amt_sum_12m"):
            suffix = f" ≈ {s}" if s is not None else ""
            msgs.append(f"Total claimed amount in the last 12 months{suffix} "
                        f"{'is high and raises' if raises else 'is low and lowers'} risk.")
        elif base.startswith("amt_mean"):
            suffix = f" ≈ {a}" if a is not None else ""
            msgs.append(f"Average historical claim size{suffix} "
                        f"{'is high and raises' if raises else 'is low and lowers'} risk.")
        elif base.startswith("amt_max"):
            suffix = f" ≈ {m}" if m is not None else ""
            msgs.append(f"Maximum historical claim{suffix} "
                        f"{'is high and raises' if raises else 'is low and lowers'} risk.")
        elif base.startswith("Premium"):
            msgs.append(f"Policy premium has a "
                        f"{'positive' if raises else 'negative'} effect on risk.")
        elif base.startswith("Deductible"):
            msgs.append(f"Deductible level is "
                        f"{'unfavourable (tends to raise risk)' if raises else 'favourable (tends to lower risk)'}."
                       )
        elif base.startswith("Species_"):
            sp = base.replace("Species_", "")
            msgs.append(f"Species ({sp}) shows "
                        f"{'higher' if raises else 'lower'} risk in the portfolio.")
        elif base.startswith("Breed_"):
            br = base.replace("Breed_", "")
            msgs.append(f"Breed ({br}) shows "
                        f"{'higher' if raises else 'lower'} risk in the portfolio.")
        elif base.startswith("loss_ratio_12m"):
            msgs.append(f"Paid-to-premium ratio over 12 months "
                        f"{'is high' if raises else 'is low'}, "
                        f"{'raising' if raises else 'lowering'} risk.")

        elif base.startswith("pet_age_months"):
            age = getattr(payload, "pet_age_months", None)
            if age is not None:
                yrs = round(float(age)/12.0, 1)
                msgs.append(f"Pet age (~{yrs} years) "
                            f"{'is higher and raises' if raises else 'is lower and lowers'} risk.")
            else:
                msgs.append(f"Pet age contributes and "
                            f"{'raises' if raises else 'lowers'} risk.")

        if len(msgs) >= 3:
            break

    if not msgs:
        msgs = ["The model combines multiple historical signals to produce the recommendation."]
    return " ".join(msgs)


# ==== Endpoints ====
@app.post("/predict", response_model=DecisionOutput)
def predict(payload: ClaimInput, model_key: Optional[str] = None):
    # 1) Load latest model + meta
    mk = _select_model_key(payload, model_key)
    pipe, meta = _load_model_or_http(mk)
    cols = meta["num_cols"] + meta["cat_cols"]

    # 2) Build one-row dataframe
    row_dict = {c: None for c in cols}
    for k, v in payload.model_dump(exclude_none=True).items():
        if k in row_dict:
            row_dict[k] = v
    X = pd.DataFrame([row_dict], columns=cols)

    age_penalty = _apply_age_handling(payload, X)

    # 3) Predict -> 风险概率 = 1 - p(approve)
    proba = pipe.predict_proba(X)[0]
    # 按训练约定：Decision==1 表示“auto-approve 倾向”
    p_approve = float(proba[1])
    p_risky = 1.0 - p_approve

    if age_penalty > 0:
        p_risky = min(0.99, p_risky + age_penalty)

    # 为了 UI 好看做一点夹紧
    p_risky = max(0.01, min(0.99, p_risky))

    # label 用和 /score 一样的三档规则
    label = _map_prob_to_label(p_risky)

    # 4) Explain + rationale
    try:
        topk = explain(pipe, X)   # 返回 [(feature_name, shap_or_importance), ...]
    except Exception:
        topk = []
    rationale = _nl_explain_en(topk, payload)

    return DecisionOutput(
        label=label,
        score=p_risky,
        top_features=topk,
        rationale=rationale
    )


@app.post("/score", response_model=ScoreResponse)
def score(payload: ClaimInput, model_key: Optional[str] = None):
    mk = _select_model_key(payload, model_key)
    pipe, meta = _load_model_or_http(mk)
    cols = meta["num_cols"] + meta["cat_cols"]

    row = {c: None for c in cols}
    for k, v in payload.model_dump(exclude_none=True).items():
        if k in row:
            row[k] = v
    X = pd.DataFrame([row])

    age_penalty = _apply_age_handling(payload, X)

    p_approve = float(pipe.predict_proba(X)[0][1])  # class 1 = approve tendency (per training)
    p_risky = 1.0 - p_approve                       # we display risky prob in UI

    if age_penalty > 0:
        p_risky = min(0.99, p_risky + age_penalty)

    label = _map_prob_to_label(p_risky)

    return ScoreResponse(
        label=label,
        score=p_risky,
        model_version=meta.get("model_version", "unknown")
    )

@app.post("/explain", response_model=ExplainResponse)
def explain_route(payload: ClaimInput, model_key: Optional[str] = None):
    # 1) 载入最新模型与元数据
    mk = _select_model_key(payload, model_key)
    pipe, meta = _load_model_or_http(mk)
    cols = meta["num_cols"] + meta["cat_cols"]

    # 2) 组一行与训练一致的列；缺失留 None，由流水线里的 Imputer 处理
    row = {c: None for c in cols}
    for k, v in payload.model_dump(exclude_none=True).items():
        if k in row:
            row[k] = v
    X = pd.DataFrame([row])

    # Apply age handling for consistency
    _apply_age_handling(payload, X)

    # 3) 调用统一版解释函数（返回 top-k (name, shap_value)）
    topk = explain(pipe, X)

    # 4) 映射到你的响应结构（importance=|贡献值|, contribution=原始SHAP）
    top_features = [
        TopFeature(name=name, importance=abs(val), contribution=float(val))
        for name, val in topk
    ]

    return ExplainResponse(
        top_features=top_features,
        method="shap",
        stability_score=None,
        plot_uri=None
    )

@app.post("/decisions/override", response_model=OverrideResponse)
def override(req: OverrideRequest):
    # Disabled in this demo build to avoid undefined globals (PIPE/MODEL_VERSION).
    raise HTTPException(status_code=501, detail="Override endpoint is temporarily disabled in this demo build.")

@app.get("/options")
def options():
    return _load_options()

app.include_router(admin)

# # ---- 极简演示 UI ----
UI_DIR = Path(__file__).resolve().parent.parent / "web"

@app.get("/", response_class=HTMLResponse)
def ui():
    return FileResponse(UI_DIR / "index.html")

@app.get("/meta")
def meta(model_key: str = "underwriting"):
    try:
        _, meta = _load_model_or_http(model_key)
        ver = str(meta.get("model_version", "untrained"))
    except Exception:
        ver = "untrained"
    return {"model_version": ver, "model_key": model_key}