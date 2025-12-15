# app/services/model_train.py
from pathlib import Path
import pandas as pd, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
    # 这里有两个不同的目标：
    # - 训练标签(Decision)可以来自历史理赔(弱标签)
    # - 但线上推理时，往往只拿得到“投保/宠物静态信息”(quote-time features)
    # 为了让 /predict 能根据输入的宠物数据产生差异化结果，
    # 我们把模型特征限制为 quote-time 可用列。
    ALL_NUM = [
        "Premium",
        "Deductible",
        "pet_age_months",
    ]
    ALL_CAT = [
        "Species",
        "Breed",
        "EnrollPath",
        "Sex",
        "Country",
    ]


    num_cols = [c for c in ALL_NUM if c in df.columns]
    # Some derived columns can end up entirely missing depending on data quality.
    # Drop all-missing numeric columns to avoid imputer warnings and keep the pipeline stable.
    num_cols = [c for c in num_cols if df[c].notna().any()]
    cat_cols = [c for c in ALL_CAT if c in df.columns]

    # 至少要有一些可用特征，否则训练没有意义
    if len(num_cols) + len(cat_cols) == 0:
        raise ValueError(f"No usable features found for training. Columns in data: {df.columns.tolist()}")

    X = df[num_cols + cat_cols].copy()
    y = df["Decision"]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(out_dir, "model.pkl"))
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "model_version": "v0.3-quote-features-lr"}
    Path(out_dir, "model_meta.json").write_text(json.dumps(meta))

    print("Train acc:", pipe.score(Xtr, ytr), " Test acc:", pipe.score(Xte, yte))
    print("Using num_cols:", num_cols, " cat_cols:", cat_cols)
    return pipe


def train_claims(data_dir="data", out_dir="models"):
    df = load_features(data_dir)

    ALL_NUM = [
        "amt_sum_12m",
        "amt_mean",
        "amt_max",
        "n_claims",
        "loss_ratio_12m",
        "days_since_policy_start",
        "preexisting_proxy",
        "Premium",
        "Deductible",
        "pet_age_months",
    ]
    ALL_CAT = [
        "Species",
        "Breed",
        "EnrollPath",
        "Sex",
        "Country",
    ]

    num_cols = [c for c in ALL_NUM if c in df.columns]
    # Some derived columns can end up entirely missing depending on data quality.
    # Drop all-missing numeric columns to avoid imputer warnings and keep the pipeline stable.
    num_cols = [c for c in num_cols if df[c].notna().any()]
    cat_cols = [c for c in ALL_CAT if c in df.columns]

    if not any(c in num_cols for c in ["amt_sum_12m", "amt_mean", "amt_max", "n_claims"]):
        raise ValueError(f"Core claim-history features missing. Columns in data: {df.columns.tolist()}")

    X = df[num_cols + cat_cols].copy()
    y = df["Decision"]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    clf = LogisticRegression(
        max_iter=3000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, Path(out_dir, "model_claims.pkl"))
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "model_version": "v0.1-claims-risk-lr"}
    Path(out_dir, "model_claims_meta.json").write_text(json.dumps(meta))

    print("Claims model Train acc:", pipe.score(Xtr, ytr), " Test acc:", pipe.score(Xte, yte))
    print("Claims model Using num_cols:", num_cols, " cat_cols:", cat_cols)
    return pipe
