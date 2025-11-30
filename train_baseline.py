# train_baseline.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

np.random.seed(123)

# --- 合成一批训练样本（两列特征：claim_amount, pet_age_months） ---
N = 1500
claim_amount = np.random.uniform(50, 2000, N)         # 理赔金额
pet_age = np.random.randint(1, 180, N)               # 宠物年龄（月）

# 规则生成“风险分”，并据此打标签（1=高风险(拒绝)，0=低风险(通过)）
risk_score = 0.002 * claim_amount + 0.01 * pet_age + np.random.normal(0, 0.1, N)
y = (risk_score > 2.0).astype(int)  # 阈值可随便定，够用就好

X = pd.DataFrame({
    'claim_amount': claim_amount,
    'pet_age_months': pet_age
})

# --- 简单管道：标准化 + 逻辑回归 ---
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

pipe.fit(X, y)

Path("models").mkdir(exist_ok=True)
artifact = {
    'pipeline': pipe,
    'feature_order': ['claim_amount', 'pet_age_months'],
    'model_version': 'lr_2025_11_03'
}
dump(artifact, "models/model.joblib")
print("✅ Saved models/model.joblib")
