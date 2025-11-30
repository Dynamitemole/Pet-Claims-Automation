# app/schemas.py
from pydantic import BaseModel
from typing import Optional

class ClaimInput(BaseModel):
    Species: Optional[str] = None
    Breed: Optional[str] = None
    Sex: Optional[str] = None
    Country: Optional[str] = None
    EnrollPath: Optional[str] = None
    amt_sum_12m: Optional[float] = None
    amt_mean: Optional[float] = None
    amt_max: Optional[float] = None
    n_claims: Optional[int] = None
    Premium: Optional[float] = None
    Deductible: Optional[float] = None
    pet_age_months: Optional[float] = None   # NEW
    # preexisting_conditions: Optional[int] = None  # 留空，原始数据没有

class DecisionOutput(BaseModel):
    label: str
    score: float
    top_features: list[tuple[str, float]]
    rationale: str

