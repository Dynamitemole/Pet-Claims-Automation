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
SAMPLE_CSV = Path("data/sample_claims.csv")
DECISIONS_CSV = Path("data/decisions.csv")
AUDIT_CSV = Path("data/audit_log.csv")

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

# --- CSV 懒加载（编码兜底） ---
def load_sample_df() -> pd.DataFrame:
    if not SAMPLE_CSV.exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp1252", "latin1"):
        try:
            return pd.read_csv(SAMPLE_CSV, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    return pd.read_csv(SAMPLE_CSV, encoding="latin1", on_bad_lines="skip")

# 背景数据：优先 CSV 抽样；否则合成
def _background() -> np.ndarray:
    df = load_sample_df()
    if not df.empty and all(col in df.columns for col in FEATURE_ORDER):
        bg = df[FEATURE_ORDER].copy()
        if len(bg) > 50:
            bg = bg.sample(50, random_state=0)
        return bg.values
    rng_claim = np.linspace(50, 2000, 20)
    rng_age = np.linspace(1, 180, 20)
    grid = np.array(np.meshgrid(rng_claim, rng_age)).T.reshape(-1, 2)
    bg = pd.DataFrame(grid, columns=FEATURE_ORDER).sample(50, random_state=0)
    return bg.values

def _features_from_claim_id(claim_id: str) -> Features:
    df = load_sample_df()
    if df.empty:
        raise HTTPException(status_code=400, detail="sample_claims.csv not found or unreadable; please pass 'features' directly")
    row = df.loc[df["claim_id"] == claim_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"claim_id '{claim_id}' not found in sample_claims.csv")
    r = row.iloc[0]
    return Features(
        claim_amount=float(r["claim_amount"]),
        pet_age_months=int(r["pet_age_months"]),
        text=str(r.get("text", ""))
    )

def _to_dataframe(feat: Features) -> pd.DataFrame:
    data = {"claim_amount": [feat.claim_amount], "pet_age_months": [feat.pet_age_months]}
    return pd.DataFrame(data)[FEATURE_ORDER]

def _map_prob_to_label(p_risky: float) -> str:
    if p_risky < 0.30:
        return "approve"
    elif p_risky > 0.70:
        return "reject"
    else:
        return "review"

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
def predict(payload: ClaimInput):
    # 1) Load latest model + meta
    pipe, meta = load_model()
    cols = meta["num_cols"] + meta["cat_cols"]

    # 2) Build one-row dataframe
    row_dict = {c: None for c in cols}
    for k, v in payload.model_dump(exclude_none=True).items():
        if k in row_dict:
            row_dict[k] = v
    X = pd.DataFrame([row_dict], columns=cols)

    # 3) Predict -> 风险概率 = 1 - p(approve)
    proba = pipe.predict_proba(X)[0]
    # 按训练约定：Decision==1 表示“auto-approve 倾向”
    p_approve = float(proba[1])
    p_risky = 1.0 - p_approve

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
def score(payload: ClaimInput):
    pipe, meta = load_model()
    cols = meta["num_cols"] + meta["cat_cols"]

    row = {c: None for c in cols}
    for k, v in payload.model_dump(exclude_none=True).items():
        if k in row:
            row[k] = v
    X = pd.DataFrame([row])

    p_approve = float(pipe.predict_proba(X)[0][1])  # class 1 = approve tendency (per training)
    p_risky = 1.0 - p_approve                       # we display risky prob in UI
    label = _map_prob_to_label(p_risky)

    return ScoreResponse(
        label=label,
        score=p_risky,
        model_version=meta.get("model_version", "unknown")
    )

@app.post("/explain", response_model=ExplainResponse)
def explain_route(payload: ClaimInput):
    # 1) 载入最新模型与元数据
    pipe, meta = load_model()
    cols = meta["num_cols"] + meta["cat_cols"]

    # 2) 组一行与训练一致的列；缺失留 None，由流水线里的 Imputer 处理
    row = {c: None for c in cols}
    for k, v in payload.model_dump(exclude_none=True).items():
        if k in row:
            row[k] = v
    X = pd.DataFrame([row])

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


app.include_router(admin)

# # ---- 极简演示 UI ----
UI_DIR = Path(__file__).resolve().parent.parent / "web"

@app.get("/", response_class=HTMLResponse)
def ui():
    return FileResponse(UI_DIR / "index.html")

@app.get("/meta")
def meta():
    try:
        _, meta = load_model()
        ver = str(meta.get("model_version", "untrained"))
    except Exception:
        ver = "untrained"
    return {"model_version": ver}