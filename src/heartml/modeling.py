"""EDA, cross-validated model comparison, MLflow tracking, and model persistence."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from heartml.data import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
                          load_uci_cleveland, validate_dataset)

RANDOM_STATE = 42

def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES),
                              ("categorical", categorical, CATEGORICAL_FEATURES)])

def build_models() -> dict:
    return {
        "logistic_regression": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([
            ("preprocess", make_preprocessor()),
            ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])
    }

def create_eda_artifacts(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_title("Class balance (0 = absent, 1 = present)")
    p = output_dir / "class_balance.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); paths.append(p)

    df[FEATURES].hist(figsize=(12, 9), bins=15)
    plt.suptitle("Feature distributions", y=1.01)
    p = output_dir / "feature_histograms.png"
    plt.tight_layout(); plt.savefig(p, dpi=140, bbox_inches="tight"); plt.close("all"); paths.append(p)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(df[FEATURES + [TARGET]].corr(numeric_only=True), cmap="coolwarm",
                center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    p = output_dir / "correlation_heatmap.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig); paths.append(p)
    return paths

def train_and_track(df: pd.DataFrame | None = None, tracking_uri: str | None = None,
                    output_dir: str | Path = "artifacts") -> dict:
    """Train from an in-memory DataFrame. If omitted, fetch UCI data into memory."""
    if df is None:
        df = load_uci_cleveland()
    df = validate_dataset(df)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    plot_paths = create_eda_artifacts(df, out)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-classification")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy":"accuracy", "precision":"precision", "recall":"recall", "roc_auc":"roc_auc"}
    summary, best_name, best_model, best_auc = [], None, None, -np.inf

    for name, model in build_models().items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"model":name, "random_state":RANDOM_STATE, "test_size":0.2,
                               "cv_folds":5, "dataset":"UCI Heart Disease - Cleveland"})
            cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
            for metric in scoring:
                mlflow.log_metric(f"cv_{metric}_mean", float(np.mean(cv_results[f"test_{metric}"])))
                mlflow.log_metric(f"cv_{metric}_std", float(np.std(cv_results[f"test_{metric}"])))
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
            metrics = {"test_accuracy":accuracy_score(y_test,pred),
                       "test_precision":precision_score(y_test,pred,zero_division=0),
                       "test_recall":recall_score(y_test,pred,zero_division=0),
                       "test_roc_auc":roc_auc_score(y_test,prob)}
            mlflow.log_metrics({k:float(v) for k,v in metrics.items()})
            mlflow.log_dict(classification_report(y_test,pred,output_dict=True,zero_division=0),
                            "classification_report.json")

            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(confusion_matrix(y_test,pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix — {name}")
            cm = out / f"{name}_confusion_matrix.png"
            fig.tight_layout(); fig.savefig(cm,dpi=140); plt.close(fig)

            fig, ax = plt.subplots(figsize=(5,4))
            RocCurveDisplay.from_predictions(y_test,prob,ax=ax); ax.set_title(f"ROC curve — {name}")
            roc = out / f"{name}_roc_curve.png"
            fig.tight_layout(); fig.savefig(roc,dpi=140); plt.close(fig)
            for artifact in plot_paths + [cm, roc]:
                mlflow.log_artifact(str(artifact))
            mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["numpy.dtype", "sklearn.compose._column_transformer._RemainderColsList", "sklearn.tree._tree.Tree",],)
            row = {"model":name, **{k:float(v) for k,v in metrics.items()},
                   "cv_roc_auc_mean":float(np.mean(cv_results["test_roc_auc"])),
                   "run_id":run.info.run_id}
            summary.append(row)
            if row["cv_roc_auc_mean"] > best_auc:
                best_name, best_model, best_auc = name, model, row["cv_roc_auc_mean"]

    model_path = out / "model.joblib"; joblib.dump(best_model, model_path)
    summary_path = out / "metrics_summary.json"
    summary_path.write_text(json.dumps({"selected_model":best_name,"results":summary},indent=2),
                            encoding="utf-8")
    print(pd.DataFrame(summary).sort_values("cv_roc_auc_mean",ascending=False).to_string(index=False))
    print(f"Selected by mean CV ROC-AUC: {best_name}")
    print(f"Saved model pipeline: {model_path.resolve()}")
    return {"selected_model":best_name,"model_path":str(model_path),"results":summary}

def main():
    parser = argparse.ArgumentParser(description="Train using in-memory UCI data; no CSV required.")
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    train_and_track(df=None, tracking_uri=args.tracking_uri, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
