"""
Module 5: Machine Learning Model Training & Evaluation
------------------------------------------------------
Trains and compares four Scikit-learn classifiers inside leak-free Pipelines:
  1. Logistic Regression
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Support Vector Machine (with probability calibration)

Computes Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelEvaluation:
    name: str
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    feature_importances: pd.DataFrame


def _extract_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """Extract feature weights or Gini importances from the fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        scores = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        scores = np.abs(classifier.coef_[0])
    else:
        scores = np.zeros(len(feature_names))

    total = scores.sum() if scores.sum() > 0 else 1.0
    normalized = (scores / total) * 100.0
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]

    return (
        pd.DataFrame({"Feature": clean_names, "Importance_Pct": np.round(normalized, 2)})
        .sort_values("Importance_Pct", ascending=False)
        .reset_index(drop=True)
    )


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor,
    random_state: int = 42,
) -> dict[str, ModelEvaluation]:
    """
    Fit Logistic Regression, Decision Tree, Random Forest, and SVM using
    separate cloned pipelines to prevent any cross-model or train-test leakage.
    """
    from sklearn.base import clone

    model_specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced", random_state=random_state
        ),
        "Support Vector Machine": SVC(
            kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=random_state
        ),
    }

    results: dict[str, ModelEvaluation] = {}

    for name, clf in model_specs.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("classifier", clf),
            ]
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = ModelEvaluation(
            name=name,
            pipeline=pipe,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_test, y_prob)),
            confusion_matrix=confusion_matrix(y_test, y_pred),
            fpr=fpr,
            tpr=tpr,
            feature_importances=_extract_feature_importances(pipe),
        )

    return results
