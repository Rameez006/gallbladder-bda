"""
Module 2: Data Preprocessing Pipeline
-------------------------------------
Cleans raw medical records, removes duplicates, imputes missing values,
encodes categorical columns, and prepares leak-free train/test splits.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingReport:
    raw_rows: int
    duplicates_removed: int
    missing_values_found: int
    cleaned_rows: int
    train_rows: int
    test_rows: int
    feature_names: list[str]


def clean_medical_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Perform dataset-level cleaning (duplicate removal, numeric coercion,
    and dropping rows missing the target label).
    Note: Imputation and scaling are fit ONLY on training data later
    to prevent data leakage.
    """
    raw_rows = len(df)
    missing_before = int(df.isna().sum().sum())

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    duplicates_removed = raw_rows - len(cleaned)

    if "Cancer_Risk" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["Cancer_Risk"]).reset_index(drop=True)
        cleaned["Cancer_Risk"] = cleaned["Cancer_Risk"].astype(int)

    summary = {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates_removed,
        "missing_values_found": missing_before,
        "cleaned_rows": len(cleaned),
    }
    return cleaned, summary


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Create a Scikit-learn ColumnTransformer that imputes and normalizes
    numeric columns and one-hot encodes categorical columns.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str = "Cancer_Risk",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Clean raw DataFrame, perform stratified train-test split, and return
    both raw splits and a fresh leak-free ColumnTransformer.
    """
    cleaned, summary = clean_medical_dataframe(df)

    drop_cols = [c for c in ["Patient_ID", target_col] if c in cleaned.columns]
    X = cleaned.drop(columns=drop_cols)
    y = cleaned[target_col]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    report = PreprocessingReport(
        raw_rows=summary["raw_rows"],
        duplicates_removed=summary["duplicates_removed"],
        missing_values_found=summary["missing_values_found"],
        cleaned_rows=summary["cleaned_rows"],
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_names=X.columns.tolist(),
    )

    return X_train, X_test, y_train, y_test, preprocessor, report
