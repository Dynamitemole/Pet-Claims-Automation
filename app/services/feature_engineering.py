# app/services/feature_engineering.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

PETS_ALIASES = {
    "PetId":      ["petid","pet_id","id","customerpetid"],
    "EnrollDate": ["enrolldate","enrollmentdate","policy_start","policystartdate","startdate"],
    "PetAge":     ["petage","pet_age","age_band"],
    "Species":    ["species","animal","pet_species","type"],
    "Breed":      ["breed","variety"],
    "AgeAtEnrollment": ["ageatenrollment","age","enrollmentage","age_months","agemonths"],
    "Premium":    ["premium","annual_premium","premiumamount"],
    "Deductible": ["deductible","excess"],
    "Sex":        ["sex","gender"],
    "PostalCode": ["postalcode","zip","zipcode","post_code"],
    "Country":    ["country","countrycode","country_code","nation"],
    "EnrollPath": ["enrollpath","channel","sales_channel"]
}
CLAIM_ALIASES = {
    "PetId":       ["petid","pet_id","id","customerpetid"],
    "ClaimId":     ["claimid","claim_id","case_id"],
    "ClaimDate":   ["claimdate","date","created","incidentdate"],
    "ClaimAmount": ["claimamount","amount","paid","payout"]
}

def rename_by_alias(df: pd.DataFrame, aliases: dict[str, list[str]], required: list[str] | None = None) -> pd.DataFrame:
    """Rename columns in df using alias dict; ensure required columns exist."""
    out = df.copy()
    lower_map = {c.lower(): c for c in out.columns}
    for std, alist in aliases.items():
        if std in out.columns:
            continue
        found = None
        for a in alist:
            if a in out.columns:
                found = a; break
            if a.lower() in lower_map:
                found = lower_map[a.lower()]; break
        if found is not None:
            out = out.rename(columns={found: std})
    if required:
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise ValueError(f"Missing required columns after alias rename: {missing}")
    return out

def read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ("utf-8","utf-8-sig","cp1252","latin1","gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="latin1")

def parse_age_to_months(s: str) -> float | None:
    if pd.isna(s): return np.nan
    t = str(s).lower()
    if "8 weeks to 12 months" in t:  # wide bucket → ~6 months for demo
        return 6.0
    if "0-7 weeks" in t:
        return 1.5
    m = re.search(r"(\d+)\s+years?", t)
    if m:
        return float(m.group(1)) * 12.0
    m = re.search(r"(\d+)\s+year", t)
    if m:
        return float(m.group(1)) * 12.0
    m = re.search(r"(\d+)\s+months?", t)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s+weeks?", t)
    if m:
        return float(m.group(1)) / 4.345
    return np.nan

def load_raw(data_dir: str | Path):
    data_dir = Path(data_dir)
    pets_raw   = read_csv_any(data_dir / "external" / "PetData.csv")
    claims_raw = read_csv_any(data_dir / "external" / "ClaimData.csv")

    pets   = rename_by_alias(pets_raw,   PETS_ALIASES,  required=["PetId"])
    claims = rename_by_alias(claims_raw, CLAIM_ALIASES, required=["PetId","ClaimDate","ClaimAmount"])

    # --- types & derived ---
    pets["EnrollDate"] = pd.to_datetime(pets.get("EnrollDate"), errors="coerce")
    # pet_age_months: prefer numeric AgeAtEnrollment; else parse free-text PetAge
    if "AgeAtEnrollment" in pets.columns:
        pets["pet_age_months"] = pd.to_numeric(pets["AgeAtEnrollment"], errors="coerce")
    elif "PetAge" in pets.columns:
        pets["pet_age_months"] = pets["PetAge"].apply(parse_age_to_months)
    else:
        pets["pet_age_months"] = np.nan

    claims["ClaimDate"] = pd.to_datetime(claims["ClaimDate"], errors="coerce")
    claims["ClaimAmount"] = (claims["ClaimAmount"].astype(str)
                             .str.replace(r"[^0-9.\\-]", "", regex=True)
                             .replace("", np.nan).astype(float))
    return pets, claims

def build_event_table(pets: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    # Join static pet info onto each claim for convenience
    keep = ["PetId","Species","Breed","Premium","Deductible","EnrollPath","pet_age_months","EnrollDate"]
    base = pets[keep].drop_duplicates("PetId")
    events = claims.merge(base, on="PetId", how="left")
    return events

def build_pet_agg(df_events: pd.DataFrame) -> pd.DataFrame:
    # last / first claim per pet
    last = (df_events.groupby("PetId")["ClaimDate"].max()
            .rename("last_claim_date").reset_index())
    first = (df_events.groupby("PetId")["ClaimDate"].min()
             .rename("first_claim_date").reset_index())

    # 12-month window ending at last claim
    w = df_events.merge(last, on="PetId", how="left")
    mask = (w["ClaimDate"] >= w["last_claim_date"] - pd.Timedelta(days=365)) & (w["ClaimDate"] <= w["last_claim_date"])
    w = w[mask].copy()

    agg = (w.groupby("PetId")
             .agg(amt_sum_12m=("ClaimAmount","sum"),
                  amt_mean=("ClaimAmount","mean"),
                  amt_max=("ClaimAmount","max"),
                  n_claims=("ClaimAmount","size"))
             .reset_index())

    # static dimensions
    keep = [c for c in ["PetId","Species","Breed","Premium","Deductible","EnrollPath","pet_age_months","EnrollDate","Country","Sex"]
            if c in df_events.columns]
    static = df_events[keep].drop_duplicates("PetId")

    feat = (agg.merge(static, on="PetId", how="left")
               .merge(last, on="PetId", how="left")
               .merge(first, on="PetId", how="left"))

    # derived insurance features
    feat["days_since_policy_start"] = (feat["last_claim_date"] - feat["EnrollDate"]).dt.days.astype("float")
    feat.loc[feat["days_since_policy_start"] < 0, "days_since_policy_start"] = np.nan

    feat["first_claim_gap_days"] = (feat["first_claim_date"] - feat["EnrollDate"]).dt.days.astype("float")
    feat.loc[feat["first_claim_gap_days"] < 0, "first_claim_gap_days"] = np.nan
    # weak-label proxy for pre-existing conditions: first claim within 30 days of policy start
    feat["preexisting_proxy"] = ((feat["first_claim_gap_days"] >= 0) & (feat["first_claim_gap_days"] <= 30)).astype("float")

    # paid-to-premium ratio over the last 12 months
    denom = pd.to_numeric(feat.get("Premium"), errors="coerce").replace(0, np.nan)
    feat["loss_ratio_12m"] = pd.to_numeric(feat["amt_sum_12m"], errors="coerce") / denom
        # ---- 简单的 outlier 剪裁，避免极端值把模型带偏 ----
    feat["n_claims"] = feat["n_claims"].clip(lower=0, upper=12)
    feat["amt_sum_12m"] = feat["amt_sum_12m"].clip(lower=0, upper=10000)
    feat["amt_mean"] = feat["amt_mean"].clip(lower=0, upper=5000)
    feat["amt_max"] = feat["amt_max"].clip(lower=0, upper=20000)
    feat["pet_age_months"] = feat["pet_age_months"].clip(lower=1, upper=240)  # 最多按 20 年算
    feat["loss_ratio_12m"] = feat["loss_ratio_12m"].clip(lower=0, upper=20)
    
    return feat

def make_datasets(data_dir="data"):
    data_dir = Path(data_dir)
    pets, claims = load_raw(data_dir)
    events = build_event_table(pets, claims)
    feats = build_pet_agg(events)
    (data_dir / "interim").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    events.to_csv(data_dir / "interim" / "events.csv", index=False)
    feats.to_csv(data_dir / "processed" / "pet_features.csv", index=False)
    return events, feats
