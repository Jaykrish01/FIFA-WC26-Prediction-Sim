from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled at runtime
    XGBClassifier = None


RESULT_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_RESULT = {value: key for key, value in RESULT_TO_CLASS.items()}
TARGET_LABELS = {0: "team_b_win", 1: "draw", 2: "team_a_win"}
TARGET_CLASS_ORDER = [0, 1, 2]
DRAW_CLASS = 1
XGB_DRAW_CLASS_WEIGHT_MULTIPLIER = 0.70
XGB_DRAW_BOOST_VALUES = np.arange(0.70, 1.01, 0.05)
LOGISTIC_CLASS_WEIGHT_OPTIONS: list[tuple[str, dict[int, float] | None]] = [
    ("unweighted", None),
    ("soft_draw", {0: 1.05, 1: 0.75, 2: 1.05}),
    ("moderate_draw", {0: 1.10, 1: 0.70, 2: 1.10}),
    ("strong_draw", {0: 1.15, 1: 0.60, 2: 1.15}),
]
LOGISTIC_DRAW_BOOST_VALUES = np.arange(0.45, 1.01, 0.05)
ENSEMBLE_LOGISTIC_WEIGHT_VALUES = np.arange(0.15, 0.86, 0.05)


@dataclass
class SplitMetadata:
    name: str
    start_date: str
    end_date: str
    rows: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare football match outcome models on the dataset built "
            "from dataset_builder.py using chronological train/validation/test splits."
        )
    )
    parser.add_argument(
        "--dataset",
        default="build/model_matches_features.csv",
        help="Path to the model dataset CSV produced by dataset_builder.py",
    )
    parser.add_argument(
        "--output-dir",
        default="build/training",
        help="Directory where splits, models, metrics, and plots will be written",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Fraction of unique match dates assigned to the training split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of unique match dates assigned to the validation split",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by stochastic models",
    )
    return parser.parse_args()


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    if "date" not in df.columns or "result_code" not in df.columns:
        raise ValueError("Dataset must include 'date' and 'result_code' columns")
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["date"]).reset_index(drop=True)
    return df


def build_time_splits(
    df: pd.DataFrame, train_ratio: float, val_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    unique_dates = np.array(sorted(df["date"].dt.normalize().unique()))
    total_dates = len(unique_dates)
    if total_dates < 3:
        raise ValueError("Need at least three unique dates for train/validation/test splits")

    train_date_count = max(1, int(total_dates * train_ratio))
    val_date_count = max(1, int(total_dates * val_ratio))
    if train_date_count + val_date_count >= total_dates:
        val_date_count = max(1, total_dates - train_date_count - 1)

    train_dates = set(unique_dates[:train_date_count])
    val_dates = set(unique_dates[train_date_count : train_date_count + val_date_count])
    test_dates = set(unique_dates[train_date_count + val_date_count :])

    train_df = df[df["date"].dt.normalize().isin(train_dates)].copy()
    val_df = df[df["date"].dt.normalize().isin(val_dates)].copy()
    test_df = df[df["date"].dt.normalize().isin(test_dates)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("One of the splits is empty; adjust the split ratios")

    return train_df, val_df, test_df


def feature_columns(df: pd.DataFrame) -> list[str]:
    leak_columns = {
        "date",
        "team_a",
        "team_b",
        "tournament",
        "score_a",
        "score_b",
        "score_diff",
        "abs_score_diff",
        "total_goals",
        "result_code",
        "target_a_win",
        "target_draw",
        "target_b_win",
        "elo_delta_a_post_match",
        "elo_delta_b_post_match",
        "is_warmup",
    }
    return [column for column in df.columns if column not in leak_columns]


def split_xy(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    x = df[columns].copy()
    y = df["result_code"].map(RESULT_TO_CLASS).astype(int).copy()
    return x, y


def make_logistic_pipeline(
    random_state: int,
    class_weight: dict[int, float] | str | None = None,
) -> Pipeline:
    del random_state
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def make_random_forest_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_xgboost_pipeline(random_state: int) -> Pipeline:
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.035,
                    subsample=0.88,
                    colsample_bytree=0.86,
                    min_child_weight=4,
                    gamma=0.08,
                    reg_alpha=0.15,
                    reg_lambda=1.8,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )


def predict_probabilities(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Model does not support probability predictions")
    probabilities = model.predict_proba(x)
    if probabilities.shape[1] != len(TARGET_CLASS_ORDER):
        raise RuntimeError("Unexpected probability shape returned by model")
    return probabilities


def metric_bundle(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    actual_draw_rate = float(np.mean(y_true_array == DRAW_CLASS))
    predicted_draw_rate = float(np.mean(np.asarray(y_pred) == DRAW_CLASS))
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "f1_macro": float(f1_score(y_true_array, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true_array, y_pred, average="weighted")),
        "draw_f1": float(
            f1_score((y_true_array == DRAW_CLASS).astype(int), (y_pred == DRAW_CLASS).astype(int))
        ),
        "draw_recall": float(
            recall_score((y_true_array == DRAW_CLASS).astype(int), (y_pred == DRAW_CLASS).astype(int))
        ),
        "actual_draw_rate": actual_draw_rate,
        "predicted_draw_rate": predicted_draw_rate,
        "draw_rate_error": float(abs(predicted_draw_rate - actual_draw_rate)),
        "log_loss": float(log_loss(y_true, y_proba, labels=TARGET_CLASS_ORDER)),
    }


def apply_draw_boost(y_proba: np.ndarray, boost: float) -> np.ndarray:
    adjusted = y_proba.copy()
    adjusted[:, DRAW_CLASS] *= boost
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def selection_score(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        -metrics["draw_rate_error"],
        metrics["f1_macro"],
    )


def tune_draw_boost(
    y_true: pd.Series,
    y_proba: np.ndarray,
    boost_values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    best_boost = 1.0
    best_proba = y_proba.copy()
    best_pred = np.argmax(best_proba, axis=1)
    best_metrics = metric_bundle(y_true, best_pred, best_proba)
    best_score = selection_score(best_metrics)

    for boost in boost_values:
        adjusted = apply_draw_boost(y_proba, float(boost))
        pred = np.argmax(adjusted, axis=1)
        metrics = metric_bundle(y_true, pred, adjusted)
        score = selection_score(metrics)
        if score > best_score:
            best_boost = float(boost)
            best_proba = adjusted
            best_pred = pred
            best_metrics = metrics
            best_score = score

    return best_boost, best_proba, best_pred, best_metrics


def tune_probability_outputs(
    model_name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    boost_values: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    if model_name in {"xgboost", "logistic_regression"}:
        return tune_draw_boost(y_true, y_proba, boost_values)

    pred = np.argmax(y_proba, axis=1)
    metrics = metric_bundle(y_true, pred, y_proba)
    return 1.0, y_proba, pred, metrics


def write_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    labels = [TARGET_LABELS[value] for value in TARGET_CLASS_ORDER]
    matrix = confusion_matrix(y_true, y_pred, labels=TARGET_CLASS_ORDER)

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                str(matrix[row_idx, col_idx]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_CLASS_ORDER,
        target_names=[TARGET_LABELS[value] for value in TARGET_CLASS_ORDER],
        output_dict=True,
        zero_division=0,
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def prediction_frame(
    source_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    source_df = source_df.reset_index(drop=True)
    y_true_values = pd.Series(np.asarray(y_true), index=source_df.index)
    y_pred_values = pd.Series(np.asarray(y_pred), index=source_df.index)

    frame = pd.DataFrame(
        {
            "date": source_df["date"].dt.strftime("%Y-%m-%d"),
            "team_a": source_df["team_a"],
            "team_b": source_df["team_b"],
            "tournament": source_df["tournament"],
            "actual_class": y_true_values,
            "actual_result_code": y_true_values.map(CLASS_TO_RESULT),
            "actual_label": y_true_values.map(TARGET_LABELS),
            "predicted_class": y_pred_values,
            "predicted_result_code": y_pred_values.map(CLASS_TO_RESULT),
            "predicted_label": y_pred_values.map(TARGET_LABELS),
        }
    )
    for class_index, label in TARGET_LABELS.items():
        frame[f"proba_{label}"] = y_proba[:, class_index]
    return frame


def feature_importance_frame(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        return pd.DataFrame(
            {"feature": feature_names, "importance": values}
        ).sort_values("importance", ascending=False)

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coef), axis=0)
        return pd.DataFrame(
            {"feature": feature_names, "importance": values}
        ).sort_values("importance", ascending=False)

    return pd.DataFrame(columns=["feature", "importance"])


def split_metadata(name: str, df: pd.DataFrame) -> SplitMetadata:
    return SplitMetadata(
        name=name,
        start_date=df["date"].min().strftime("%Y-%m-%d"),
        end_date=df["date"].max().strftime("%Y-%m-%d"),
        rows=int(len(df)),
    )


def fit_and_tune_model(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["model__sample_weight"] = sample_weight
    model.fit(x_train, y_train, **fit_kwargs)

    val_proba = predict_probabilities(model, x_val)
    test_proba = predict_probabilities(model, x_test)

    draw_boost = 1.0
    if model_name == "xgboost":
        draw_boost, val_proba, val_pred, val_metrics = tune_draw_boost(
            y_val,
            val_proba,
            XGB_DRAW_BOOST_VALUES,
        )
        test_proba = apply_draw_boost(test_proba, draw_boost)
        test_pred = np.argmax(test_proba, axis=1)
    elif model_name == "logistic_regression":
        draw_boost, val_proba, val_pred, val_metrics = tune_draw_boost(
            y_val,
            val_proba,
            LOGISTIC_DRAW_BOOST_VALUES,
        )
        test_proba = apply_draw_boost(test_proba, draw_boost)
        test_pred = np.argmax(test_proba, axis=1)
    else:
        val_pred = np.argmax(val_proba, axis=1)
        test_pred = np.argmax(test_proba, axis=1)
        val_metrics = metric_bundle(y_val, val_pred, val_proba)
    test_metrics = metric_bundle(y_test, test_pred, test_proba)

    return {
        "model": model,
        "draw_boost": draw_boost,
        "validation_proba": val_proba,
        "validation_pred": val_pred,
        "validation_metrics": val_metrics,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
    }


def save_model_artifacts(
    model_name: str,
    model_dir: Path,
    fitted_model: Pipeline | None,
    feature_names: list[str] | None,
    val_source_df: pd.DataFrame,
    y_val: pd.Series,
    val_proba: np.ndarray,
    val_pred: np.ndarray,
    test_source_df: pd.DataFrame,
    y_test: pd.Series,
    test_proba: np.ndarray,
    test_pred: np.ndarray,
    draw_boost: float,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    if fitted_model is not None:
        joblib.dump(fitted_model, model_dir / "model.joblib")
    write_confusion_matrix(
        y_true=y_val,
        y_pred=val_pred,
        output_path=model_dir / "confusion_matrix_validation.png",
        title=f"{model_name} Validation Confusion Matrix",
    )
    write_confusion_matrix(
        y_true=y_test,
        y_pred=test_pred,
        output_path=model_dir / "confusion_matrix_test.png",
        title=f"{model_name} Test Confusion Matrix",
    )
    val_report = write_classification_report(
        y_true=y_val,
        y_pred=val_pred,
        output_path=model_dir / "classification_report_validation.json",
    )
    test_report = write_classification_report(
        y_true=y_test,
        y_pred=test_pred,
        output_path=model_dir / "classification_report_test.json",
    )
    prediction_frame(
        source_df=val_source_df,
        y_true=y_val,
        y_pred=val_pred,
        y_proba=val_proba,
    ).to_csv(model_dir / "validation_predictions.csv", index=False)
    prediction_frame(
        source_df=test_source_df,
        y_true=y_test,
        y_pred=test_pred,
        y_proba=test_proba,
    ).to_csv(model_dir / "test_predictions.csv", index=False)

    if fitted_model is not None and feature_names is not None:
        importance_df = feature_importance_frame(fitted_model, feature_names)
    else:
        importance_df = pd.DataFrame(columns=["feature", "importance"])
    importance_path = model_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    val_metrics = metric_bundle(y_val, val_pred, val_proba)
    test_metrics = metric_bundle(y_test, test_pred, test_proba)

    result = {
        "model_name": model_name,
        "draw_boost": draw_boost,
        "validation": val_metrics,
        "test": test_metrics,
        "validation_report": val_report,
        "test_report": test_report,
        "artifacts": {
            "model": str(model_dir / "model.joblib"),
            "validation_confusion_matrix": str(model_dir / "confusion_matrix_validation.png"),
            "test_confusion_matrix": str(model_dir / "confusion_matrix_test.png"),
            "validation_report": str(model_dir / "classification_report_validation.json"),
            "test_report": str(model_dir / "classification_report_test.json"),
            "feature_importance": str(importance_path),
            "validation_predictions": str(model_dir / "validation_predictions.csv"),
            "test_predictions": str(model_dir / "test_predictions.csv"),
        },
    }
    if extra_metadata:
        result.update(extra_metadata)
    (model_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    val_source_df: pd.DataFrame,
    test_source_df: pd.DataFrame,
    feature_names: list[str],
    model_dir: Path,
    sample_weight: np.ndarray | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tuned = fit_and_tune_model(
        model_name=model_name,
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        sample_weight=sample_weight,
    )

    return save_model_artifacts(
        model_name=model_name,
        model_dir=model_dir,
        fitted_model=tuned["model"],
        feature_names=feature_names,
        val_source_df=val_source_df,
        y_val=y_val,
        val_proba=tuned["validation_proba"],
        val_pred=tuned["validation_pred"],
        test_source_df=test_source_df,
        y_test=y_test,
        test_proba=tuned["test_proba"],
        test_pred=tuned["test_pred"],
        draw_boost=tuned["draw_boost"],
        extra_metadata=extra_metadata,
    )


def blend_probabilities(primary: np.ndarray, secondary: np.ndarray, primary_weight: float) -> np.ndarray:
    blended = primary_weight * primary + (1.0 - primary_weight) * secondary
    blended /= blended.sum(axis=1, keepdims=True)
    return blended


def search_ensemble_weight(
    y_true: pd.Series,
    logistic_proba: np.ndarray,
    xgb_proba: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    best_weight = 0.5
    best_proba = blend_probabilities(logistic_proba, xgb_proba, best_weight)
    best_pred = np.argmax(best_proba, axis=1)
    best_metrics = metric_bundle(y_true, best_pred, best_proba)
    best_score = selection_score(best_metrics)

    for logistic_weight in ENSEMBLE_LOGISTIC_WEIGHT_VALUES:
        ensemble_proba = blend_probabilities(logistic_proba, xgb_proba, float(logistic_weight))
        pred = np.argmax(ensemble_proba, axis=1)
        metrics = metric_bundle(y_true, pred, ensemble_proba)
        score = selection_score(metrics)
        if score > best_score:
            best_weight = float(logistic_weight)
            best_proba = ensemble_proba
            best_pred = pred
            best_metrics = metrics
            best_score = score

    return best_weight, best_proba, best_pred, best_metrics


def evaluate_logistic_candidate(
    class_weight_name: str,
    class_weight: dict[int, float] | str | None,
    random_state: int,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    model = make_logistic_pipeline(random_state, class_weight=class_weight)
    tuned = fit_and_tune_model(
        model_name="logistic_regression",
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
    tuned["class_weight_name"] = class_weight_name
    tuned["class_weight"] = class_weight
    return tuned


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"
    models_dir = output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    train_df, val_df, test_df = build_time_splits(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "validation.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    features = feature_columns(df)
    x_train, y_train = split_xy(train_df, features)
    x_val, y_val = split_xy(val_df, features)
    x_test, y_test = split_xy(test_df, features)

    results: list[dict[str, Any]] = []
    xgb_fit: dict[str, Any] | None = None
    if XGBClassifier is not None:
        xgb_model = make_xgboost_pipeline(args.random_state)
        class_counts = np.bincount(y_train, minlength=len(TARGET_CLASS_ORDER))
        base_class_weight = {
            cls: float(len(y_train)) / (len(TARGET_CLASS_ORDER) * max(class_counts[cls], 1))
            for cls in TARGET_CLASS_ORDER
        }
        xgb_sample_weight = np.array(
            [
                base_class_weight[int(cls)]
                * (XGB_DRAW_CLASS_WEIGHT_MULTIPLIER if int(cls) == DRAW_CLASS else 1.0)
                for cls in y_train
            ],
            dtype=float,
        )
        xgb_fit = fit_and_tune_model(
            model_name="xgboost",
            model=xgb_model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            sample_weight=xgb_sample_weight,
        )
        xgb_result = save_model_artifacts(
            model_name="xgboost",
            model_dir=models_dir / "xgboost",
            fitted_model=xgb_fit["model"],
            feature_names=features,
            val_source_df=val_df,
            y_val=y_val,
            val_proba=xgb_fit["validation_proba"],
            val_pred=xgb_fit["validation_pred"],
            test_source_df=test_df,
            y_test=y_test,
            test_proba=xgb_fit["test_proba"],
            test_pred=xgb_fit["test_pred"],
            draw_boost=xgb_fit["draw_boost"],
            extra_metadata={
                "sample_weight_strategy": "inverse_frequency_with_draw_multiplier",
                "draw_boost_search_values": XGB_DRAW_BOOST_VALUES.tolist(),
            },
        )
        results.append(xgb_result)

    random_forest_result = evaluate_model(
        model_name="random_forest",
        model=make_random_forest_pipeline(args.random_state),
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        val_source_df=val_df,
        test_source_df=test_df,
        feature_names=features,
        model_dir=models_dir / "random_forest",
        extra_metadata={"benchmark_only": True},
    )
    results.append(random_forest_result)

    logistic_candidates: list[dict[str, Any]] = []
    for class_weight_name, class_weight in LOGISTIC_CLASS_WEIGHT_OPTIONS:
        candidate = evaluate_logistic_candidate(
            class_weight_name=class_weight_name,
            class_weight=class_weight,
            random_state=args.random_state,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
        )

        if xgb_fit is not None:
            ensemble_weight, ensemble_val_proba, ensemble_val_pred, ensemble_val_metrics = search_ensemble_weight(
                y_true=y_val,
                logistic_proba=candidate["validation_proba"],
                xgb_proba=xgb_fit["validation_proba"],
            )
            ensemble_test_proba = blend_probabilities(
                candidate["test_proba"],
                xgb_fit["test_proba"],
                ensemble_weight,
            )
            ensemble_test_pred = np.argmax(ensemble_test_proba, axis=1)
            ensemble_test_metrics = metric_bundle(y_test, ensemble_test_pred, ensemble_test_proba)
            candidate["ensemble_weight"] = ensemble_weight
            candidate["ensemble_validation_proba"] = ensemble_val_proba
            candidate["ensemble_validation_pred"] = ensemble_val_pred
            candidate["ensemble_validation_metrics"] = ensemble_val_metrics
            candidate["ensemble_test_proba"] = ensemble_test_proba
            candidate["ensemble_test_pred"] = ensemble_test_pred
            candidate["ensemble_test_metrics"] = ensemble_test_metrics
        logistic_candidates.append(candidate)

    if not logistic_candidates:
        raise RuntimeError("No logistic regression candidates were evaluated")

    def candidate_score(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
        if xgb_fit is not None and "ensemble_validation_metrics" in candidate:
            ensemble_metrics = candidate["ensemble_validation_metrics"]
            logistic_metrics = candidate["validation_metrics"]
            return selection_score(ensemble_metrics) + selection_score(logistic_metrics)
        return selection_score(candidate["validation_metrics"])

    best_logistic_candidate = max(logistic_candidates, key=candidate_score)

    logistic_result = save_model_artifacts(
        model_name="logistic_regression",
        model_dir=models_dir / "logistic_regression",
        fitted_model=best_logistic_candidate["model"],
        feature_names=features,
        val_source_df=val_df,
        y_val=y_val,
        val_proba=best_logistic_candidate["validation_proba"],
        val_pred=best_logistic_candidate["validation_pred"],
        test_source_df=test_df,
        y_test=y_test,
        test_proba=best_logistic_candidate["test_proba"],
        test_pred=best_logistic_candidate["test_pred"],
        draw_boost=best_logistic_candidate["draw_boost"],
        extra_metadata={
            "class_weight_name": best_logistic_candidate["class_weight_name"],
            "class_weight": best_logistic_candidate["class_weight"],
            "ensemble_weight": best_logistic_candidate.get("ensemble_weight"),
            "draw_boost_search_values": LOGISTIC_DRAW_BOOST_VALUES.tolist(),
        },
    )
    results.append(logistic_result)

    ensemble_result: dict[str, Any] | None = None
    if xgb_fit is not None and "ensemble_validation_metrics" in best_logistic_candidate:
        ensemble_result = save_model_artifacts(
            model_name="ensemble",
            model_dir=models_dir / "ensemble",
            fitted_model=None,
            feature_names=None,
            val_source_df=val_df,
            y_val=y_val,
            val_proba=best_logistic_candidate["ensemble_validation_proba"],
            val_pred=best_logistic_candidate["ensemble_validation_pred"],
            test_source_df=test_df,
            y_test=y_test,
            test_proba=best_logistic_candidate["ensemble_test_proba"],
            test_pred=best_logistic_candidate["ensemble_test_pred"],
            draw_boost=1.0,
            extra_metadata={
                "ensemble_logistic_weight": best_logistic_candidate["ensemble_weight"],
                "ensemble_xgb_weight": float(1.0 - best_logistic_candidate["ensemble_weight"]),
                "source_models": ["logistic_regression", "xgboost"],
                "logistic_class_weight_name": best_logistic_candidate["class_weight_name"],
            },
        )
        results.append(ensemble_result)

    comparison_rows = []
    for item in results:
        row = {"model_name": item["model_name"]}
        for split_name in ("validation", "test"):
            for metric_name, metric_value in item[split_name].items():
                row[f"{split_name}_{metric_name}"] = metric_value
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["validation_accuracy", "validation_balanced_accuracy", "validation_f1_macro"],
        ascending=False,
    )
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)

    deployable_models = comparison_df[comparison_df["model_name"].isin(["logistic_regression", "xgboost", "ensemble"])]
    best_model_name = deployable_models.iloc[0]["model_name"]
    best_model_result = next(item for item in results if item["model_name"] == best_model_name)
    benchmark_best_model = comparison_df.iloc[0]["model_name"]

    summary = {
        "dataset_path": str(dataset_path),
        "feature_count": len(features),
        "class_mapping": {
            str(class_index): {
                "label": TARGET_LABELS[class_index],
                "result_code": CLASS_TO_RESULT[class_index],
            }
            for class_index in TARGET_CLASS_ORDER
        },
        "splits": [
            asdict(split_metadata("train", train_df)),
            asdict(split_metadata("validation", val_df)),
            asdict(split_metadata("test", test_df)),
        ],
        "best_model": best_model_name,
        "benchmark_best_model": benchmark_best_model,
        "best_model_draw_boost": best_model_result.get("draw_boost", 1.0),
        "best_model_validation_metrics": best_model_result["validation"],
        "best_model_test_metrics": best_model_result["test"],
        "logistic_class_weight_name": best_logistic_candidate["class_weight_name"],
        "logistic_class_weight": best_logistic_candidate["class_weight"],
        "logistic_draw_boost": best_logistic_candidate["draw_boost"],
        "ensemble_logistic_weight": best_logistic_candidate.get("ensemble_weight"),
        "ensemble_xgb_weight": (
            None
            if best_logistic_candidate.get("ensemble_weight") is None
            else float(1.0 - best_logistic_candidate["ensemble_weight"])
        ),
        "xgboost_draw_boost": None if xgb_fit is None else xgb_fit["draw_boost"],
        "model_comparison_path": str(output_dir / "model_comparison.csv"),
    }
    if ensemble_result is not None:
        summary["ensemble_validation_metrics"] = ensemble_result["validation"]
        summary["ensemble_test_metrics"] = ensemble_result["test"]
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Features used: {len(features)}")
    print(f"Train rows: {len(train_df)} | Validation rows: {len(val_df)} | Test rows: {len(test_df)}")
    print("Model comparison:")
    print(comparison_df.to_string(index=False))
    print(f"Best deployable model: {best_model_name}")
    print(f"Best benchmark model: {benchmark_best_model}")
    print(f"Summary written to: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
