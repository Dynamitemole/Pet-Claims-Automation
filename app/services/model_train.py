# app/services/model_train.py
from pathlib import Path
import pandas as pd, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import joblib

def load_features(data_dir="data"):
    df = pd.read_csv(Path(data_dir, "processed/pet_features.csv"))

    # 若已经有人手工标过 Decision，就尊重现有列
    if "Decision" in df.columns:
        return df

    # ---- 构造弱标签：risk_index 越大风险越高 ----
    # 基础数值列，全部转成数值方便后面归一化
    n_claims = pd.to_numeric(df.get("n_claims"), errors="coerce")
    loss_ratio = pd.to_numeric(df.get("loss_ratio_12m"), errors="coerce")
    age_m = pd.to_numeric(df.get("pet_age_months"), errors="coerce")
    pre_proxy = pd.to_numeric(df.get("preexisting_proxy"), errors="coerce").fillna(0.0)

    # 归一化到 0~1 左右的尺度
    norm_n = np.clip(n_claims, 0, 6) / 6.0              # 0 次 ~ 6 次以上
    norm_lr = np.clip(loss_ratio, 0, 5) / 5.0           # loss ratio 0~5 之间
    age_years = age_m / 12.0
    norm_age = np.clip(age_years / 12.0, 0, 1.0)        # 0~12 岁映射到 0~1

    # 组合成一个简单的 risk_index
    # （权重可以根据感觉微调，但不用太纠结，主要是“多维度 + 连续”）
    risk_index = (
        0.5 * norm_lr +
        0.25 * norm_n +
        0.15 * norm_age +
        0.10 * pre_proxy
    )

    # 对 NaN 用中位数填充，防止异常值搞崩
    risk_index = risk_index.fillna(risk_index.median())

    # 取最安全的一部分作为 auto-approve
    # 例如：安全程度最高的 30% 样本标记为 1，其余为 0
    threshold = risk_index.quantile(0.30)
    df["Decision"] = (risk_index <= threshold).astype(int)

    return df


def train(data_dir="data", out_dir="models"):
    df = load_features(data_dir)

    # 所有可能的候选列（按可用性筛）
    ALL_NUM = [
    "amt_sum_12m","amt_mean","amt_max","n_claims",
    "Premium","Deductible",
    "pet_age_months","days_since_policy_start","loss_ratio_12m"
    ]
    ALL_CAT = ["Species","Breed","EnrollPath"]  # Sex/Country 不在原始CSV中，这里先不上


    num_cols = [c for c in ALL_NUM if c in df.columns]
    cat_cols = [c for c in ALL_CAT if c in df.columns]

    # 至少要有一些核心数值列
    if not any(c in num_cols for c in ["amt_sum_12m","n_claims","amt_mean","amt_max"]):
        raise ValueError(f"Core numeric features missing. Columns in data: {df.columns.tolist()}")

    X = df[num_cols + cat_cols].copy()
    y = df["Decision"]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    base = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=3,
        min_samples_leaf=50,
        l2_regularization=0.2,
        early_stopping=True,
        random_state=42,
    )
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(out_dir, "model.pkl"))
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "model_version": "v0.2-external-data"}
    Path(out_dir, "model_meta.json").write_text(json.dumps(meta))

    print("Train acc:", pipe.score(Xtr, ytr), " Test acc:", pipe.score(Xte, yte))
    print("Using num_cols:", num_cols, " cat_cols:", cat_cols)
    return pipe
