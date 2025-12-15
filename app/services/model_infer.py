# services/model_infer.py
from pathlib import Path
import json
import joblib
import shap
import numpy as np

MODELS_DIR = Path("models")

_MODEL_REGISTRY = {
    "underwriting": {"model": "model.pkl", "meta": "model_meta.json"},
    "claims": {"model": "model_claims.pkl", "meta": "model_claims_meta.json"},
}

def load_model(model_key: str = "underwriting"):
    spec = _MODEL_REGISTRY.get(model_key)
    if spec is None:
        raise FileNotFoundError(f"Unknown model_key '{model_key}'. Available: {sorted(_MODEL_REGISTRY.keys())}")

    model_path = MODELS_DIR / spec["model"]
    meta_path = MODELS_DIR / spec["meta"]
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for model_key='{model_key}'. Expected: {model_path} and {meta_path}"
        )

    pipe = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    meta["model_key"] = model_key
    return pipe, meta

def _unwrap_to_base_estimator(clf):
    """
    尝试从包装器中解包出真正的基模型（尤其是 CalibratedClassifierCV）。
    兼容多种 sklearn 版本可能的属性命名。
    """
    m = clf
    # 1) CalibratedClassifierCV 拟合后的结构
    try:
        c_list = getattr(clf, "calibrated_classifiers_", None)
        if c_list and len(c_list) > 0:
            inner = c_list[0]
            for attr in ("base_estimator", "estimator", "classifier"):
                if hasattr(inner, attr) and getattr(inner, attr) is not None:
                    return getattr(inner, attr)
    except Exception:
        pass

    # 2) 常见包装器属性
    for attr in ("best_estimator_", "estimator", "base_estimator_", "base_estimator", "classifier"):
        if hasattr(m, attr) and getattr(m, attr) is not None:
            return getattr(m, attr)

    return m

def explain(pipe, X_row):
    """
    X_row: 单行 pandas.DataFrame（列名与训练一致）
    返回: [(feature_name, contribution_value), ...] Top-5
    """
    # 0) 取出预处理与分类器
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    # 1) 预处理到纯数值矩阵
    Z = pre.transform(X_row)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    Z = np.asarray(Z, dtype=np.float64)

    # 2) 尝试对“底层树模型”做 TreeExplainer
    base_model = _unwrap_to_base_estimator(clf)

    # ---- 线性模型优先：直接用系数计算单样本贡献（稳定且无需 SHAP） ----
    # LogisticRegression: coef_ shape=(1, n_features) for binary
    if hasattr(base_model, "coef_"):
        coef = np.asarray(getattr(base_model, "coef_"), dtype=np.float64)
        if coef.ndim == 2 and coef.shape[0] >= 1 and Z.shape[0] >= 1:
            # coef 对应的是 class=1（通常是“approve/安全”）相对于 class=0 的 log-odds 贡献
            # 我们要反过来解释成“对风险”的贡献
            contrib = -(coef[0] * Z[0])
            names = pre.get_feature_names_out()
            top_idx = np.argsort(np.abs(contrib))[::-1][:5]
            return [(names[i], float(contrib[i])) for i in top_idx]

    try:
        explainer = shap.TreeExplainer(base_model)
        shap_vals = explainer.shap_values(Z)

        # binary: 旧版是 list，新版是 ndarray
        if isinstance(shap_vals, list):
            # 这里按约定：索引 1 对应“approve / 安全”这一类
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        else:
            # ndarray: shape = (n_samples, n_features)
            vals = shap_vals[0]

        # 当前 vals 是“对 approve（安全）的贡献”，
        # 我们要反过来解释成“对风险”的贡献
        vals = -np.asarray(vals, dtype=np.float64)

    except Exception:
        # 3) 兜底：若是树模型提供 feature_importances_，用它（无符号，但稳定，不会 500）
        if hasattr(base_model, "feature_importances_"):
            fi = np.asarray(base_model.feature_importances_, dtype=float)
            # 对齐维度
            if fi.shape[0] != Z.shape[1]:
                # 维度不一致就退成全 0，至少不报错
                fi = np.zeros(Z.shape[1], dtype=float)
            vals = fi
        else:
            # 最后兜底为 0 向量
            vals = np.zeros(Z.shape[1], dtype=float)

    names = pre.get_feature_names_out()
    top_idx = np.argsort(np.abs(vals))[::-1][:5]
    return [(names[i], float(vals[i])) for i in top_idx]

