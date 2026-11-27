"""
Module 1: Dataset Collection & Synthetic Educational Data Generator
-----------------------------------------------------------------
Loads a gallbladder cancer CSV dataset if supplied by the user, or
generates a clearly labeled synthetic educational dataset grounded
in clinically documented risk-factor patterns (for learning only).
"""

import os
import numpy as np
import pandas as pd

FEATURE_DESCRIPTIONS = {
    "Patient_ID": "Unique anonymized educational record identifier",
    "Age": "Patient age in years (20 to 85)",
    "Gender": "Biological sex (Female / Male) — higher incidence historically observed in females",
    "Gallstones": "Presence of cholelithiasis / gallstones (1 = Yes, 0 = No)",
    "Wall_Thickness_mm": "Gallbladder wall thickness measured in millimeters (normal < 3.0 mm)",
    "Polyp_Size_mm": "Largest gallbladder polyp diameter in mm (0 if absent)",
    "Chronic_Cholecystitis": "Long-term gallbladder inflammation (1 = Yes, 0 = No)",
    "Porcelain_Gallbladder": "Calcification of the gallbladder wall (1 = Yes, 0 = No)",
    "Family_History": "Family history of gallbladder or biliary malignancy (1 = Yes, 0 = No)",
    "BMI": "Body Mass Index in kg/m^2",
    "Bilirubin_mg_dL": "Serum total bilirubin level in mg/dL",
    "Alkaline_Phosphatase_UL": "Serum alkaline phosphatase enzyme level in U/L",
    "Jaundice": "Clinical yellowing of skin/sclera (1 = Yes, 0 = No)",
    "Abdominal_Pain": "Right upper quadrant abdominal pain (1 = Yes, 0 = No)",
    "Cancer_Risk": "Target binary label (1 = High Risk / Positive Class, 0 = Low Risk / Negative Class)",
}


def generate_synthetic_dataset(
    n_samples: int = 2400,
    seed: int = 42,
    inject_imperfections: bool = True,
) -> pd.DataFrame:
    """
    Generate a clearly labeled synthetic educational dataset for Big Data
    and Machine Learning demonstrations.

    NOTE: This data is artificially simulated for college coursework and
    MUST NOT be interpreted as real clinical trial evidence.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(54, 13, n_samples), 21, 86).round(0)
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.64, 0.36])

    # Risk factors with realistic correlations
    gallstones_prob = np.where(gender == "Female", 0.42, 0.28) + (age - 45) * 0.004
    gallstones = rng.binomial(1, np.clip(gallstones_prob, 0.1, 0.85))

    chronic_cholecystitis = rng.binomial(
        1, np.clip(0.12 + 0.35 * gallstones + (age - 50) * 0.003, 0.05, 0.75)
    )
    porcelain_gb = rng.binomial(1, np.clip(0.03 + 0.08 * chronic_cholecystitis, 0.01, 0.25))
    family_history = rng.binomial(1, 0.14, size=n_samples)
    bmi = np.clip(rng.normal(26.8 + 2.2 * gallstones, 4.4, n_samples), 16.5, 45.0).round(1)

    wall_thickness = np.clip(
        rng.normal(2.2 + 1.4 * gallstones + 1.8 * chronic_cholecystitis + 1.5 * porcelain_gb, 0.9, n_samples),
        1.0,
        12.5,
    ).round(2)

    has_polyp = rng.binomial(1, 0.18, size=n_samples)
    polyp_size = np.where(
        has_polyp == 1,
        np.clip(rng.exponential(5.5, n_samples) + 2.0, 2.0, 22.0),
        0.0,
    ).round(1)

    # Log-odds model for synthetic target generation
    logit = (
        -5.4
        + 0.042 * (age - 50)
        + 0.45 * (gender == "Female").astype(float)
        + 1.55 * gallstones
        + 0.58 * (wall_thickness - 2.5)
        + 0.16 * polyp_size
        + 1.05 * chronic_cholecystitis
        + 1.45 * porcelain_gb
        + 0.95 * family_history
        + 0.05 * (bmi - 25.0)
        + rng.normal(0, 0.65, n_samples)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    cancer_risk = (prob >= 0.5).astype(int)

    # Secondary clinical lab markers correlated with risk
    bilirubin = np.clip(
        rng.normal(0.8 + 1.6 * cancer_risk + 0.5 * gallstones, 0.6, n_samples), 0.2, 8.5
    ).round(2)
    alk_phos = np.clip(
        rng.normal(88 + 95 * cancer_risk + 28 * gallstones, 34, n_samples), 35, 480
    ).round(0)
    jaundice = rng.binomial(1, np.clip(0.06 + 0.48 * cancer_risk + 0.12 * gallstones, 0.02, 0.9))
    abdominal_pain = rng.binomial(1, np.clip(0.22 + 0.42 * gallstones + 0.25 * cancer_risk, 0.1, 0.95))

    df = pd.DataFrame(
        {
            "Patient_ID": [f"EDU-{10001 + i}" for i in range(n_samples)],
            "Age": age.astype(float),
            "Gender": gender,
            "Gallstones": gallstones.astype(float),
            "Wall_Thickness_mm": wall_thickness,
            "Polyp_Size_mm": polyp_size,
            "Chronic_Cholecystitis": chronic_cholecystitis,
            "Porcelain_Gallbladder": porcelain_gb,
            "Family_History": family_history,
            "BMI": bmi,
            "Bilirubin_mg_dL": bilirubin,
            "Alkaline_Phosphatase_UL": alk_phos,
            "Jaundice": jaundice,
            "Abdominal_Pain": abdominal_pain,
            "Cancer_Risk": cancer_risk,
        }
    )

    if inject_imperfections:
        # Inject ~2.5% missing values and ~1.5% duplicate rows so Module 2
        # demonstrates real-world data cleaning.
        for col in ["Age", "Wall_Thickness_mm", "BMI", "Bilirubin_mg_dL", "Gender"]:
            missing_idx = rng.choice(df.index, size=max(1, int(0.015 * n_samples)), replace=False)
            df.loc[missing_idx, col] = np.nan

        dup_count = max(5, int(0.015 * n_samples))
        duplicates = df.sample(n=dup_count, random_state=seed)
        df = pd.concat([df, duplicates], ignore_index=True)

    return df


def load_dataset(csv_path: str | None = None, n_samples: int = 2400) -> tuple[pd.DataFrame, str]:
    """
    Load dataset from a CSV path if provided; otherwise generate and save
    the synthetic educational dataset locally.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df, f"Loaded external CSV dataset ({len(df)} rows)"

    df = generate_synthetic_dataset(n_samples=n_samples)
    os.makedirs("data", exist_ok=True)
    default_path = os.path.join("data", "sample_gallbladder_dataset.csv")
    df.to_csv(default_path, index=False)
    return df, f"Generated Synthetic Educational Dataset ({len(df)} rows)"
