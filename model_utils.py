from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30
ALPHA = 0.05
MPG_THRESHOLD = 23.0

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "auto-mpg/auto-mpg.data"
)
COLUMNS = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "name",
]
ORIGIN_LABELS = {1: "Estados Unidos", 2: "Europa", 3: "Japón"}


@dataclass
class RegressionResult:
    model_name: str
    features: list[str]
    r2_train: float
    r2_test: float
    rmse: float
    mae: float
    intercept: float
    coefficients: dict[str, float]
    y_test: pd.Series
    predictions: np.ndarray
    residuals: np.ndarray


@dataclass
class ClassificationResult:
    solver: str
    C: float
    threshold: float
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    auc: float
    cm: np.ndarray
    y_test: pd.Series
    probabilities: np.ndarray
    predictions: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    test_weight: pd.Series


def _parse_raw_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        names=COLUMNS,
        na_values="?",
        quotechar='"',
    )


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    df = df.copy()
    for col in [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if len(df) < 50:
        raise ValueError("El conjunto cargado es demasiado pequeño para este proyecto.")
    return df


def load_auto_mpg(data_dir: str | Path = "data") -> pd.DataFrame:
    """Carga Auto MPG usando caché local y, si hace falta, UCI.

    Orden de búsqueda:
    1) data/auto_mpg.csv
    2) data/auto-mpg.data
    3) descarga desde UCI y creación de data/auto_mpg.csv
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "auto_mpg.csv"
    raw_path = data_dir / "auto-mpg.data"

    if csv_path.exists():
        return _validate(pd.read_csv(csv_path))
    if raw_path.exists():
        df = _validate(_parse_raw_file(raw_path))
        df.to_csv(csv_path, index=False)
        return df

    try:
        df = pd.read_csv(
            DATA_URL,
            sep=r"\s+",
            names=COLUMNS,
            na_values="?",
            quotechar='"',
        )
    except Exception as exc:
        raise RuntimeError(
            "No fue posible cargar Auto MPG. Conecta el equipo a Internet para la "
            "primera ejecución o guarda el archivo original como data/auto-mpg.data."
        ) from exc

    df = _validate(df)
    df.to_csv(csv_path, index=False)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["origin_label"] = out["origin"].map(ORIGIN_LABELS).fillna("Otro")
    out["efficiency_class"] = np.where(out["mpg"] > MPG_THRESHOLD, "Alta", "Baja/mediana")
    return out


def filter_data(
    df: pd.DataFrame,
    year_range: Iterable[int] | None = None,
    origins: Iterable[int] | None = None,
    cylinders: Iterable[int] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if year_range is not None:
        lo, hi = list(year_range)
        out = out[out["model_year"].between(lo, hi)]
    if origins:
        out = out[out["origin"].isin(list(origins))]
    if cylinders:
        out = out[out["cylinders"].isin(list(cylinders))]
    return out


def pearson_weight_mpg(df: pd.DataFrame) -> tuple[float, float]:
    base = df[["weight", "mpg"]].dropna()
    try:
        r, p = stats.pearsonr(base["weight"], base["mpg"], alternative="less")
    except TypeError:  # Compatibilidad con versiones antiguas de SciPy.
        r, p_two = stats.pearsonr(base["weight"], base["mpg"])
        p = p_two / 2 if r < 0 else 1 - p_two / 2
    return float(r), float(p)


def regression_result(df: pd.DataFrame, model_name: str = "simple", alpha: float = 1.0) -> RegressionResult:
    model_name = model_name.lower()
    if model_name == "simple":
        features = ["weight"]
        model = LinearRegression()
    elif model_name == "multiple":
        features = ["weight", "displacement", "acceleration"]
        model = LinearRegression()
    elif model_name == "ridge":
        features = ["weight", "displacement", "acceleration"]
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ])
    else:
        raise ValueError("model_name debe ser simple, multiple o ridge")

    base = df[features + ["mpg"]].dropna()
    X = base[features]
    y = base["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    train_pred = model.predict(X_train)

    if model_name == "ridge":
        fitted = model.named_steps["ridge"]
        intercept = float(fitted.intercept_)
        coefs = dict(zip(features, fitted.coef_.astype(float)))
    else:
        intercept = float(model.intercept_)
        coefs = dict(zip(features, model.coef_.astype(float)))

    return RegressionResult(
        model_name=model_name,
        features=features,
        r2_train=float(r2_score(y_train, train_pred)),
        r2_test=float(r2_score(y_test, pred)),
        rmse=float(mean_squared_error(y_test, pred) ** 0.5),
        mae=float(mean_absolute_error(y_test, pred)),
        intercept=intercept,
        coefficients=coefs,
        y_test=y_test,
        predictions=np.asarray(pred),
        residuals=np.asarray(y_test) - np.asarray(pred),
    )


def model_comparison(df: pd.DataFrame, ridge_alpha: float = 1.0) -> pd.DataFrame:
    rows = []
    labels = {
        "simple": "Regresión simple",
        "multiple": "Regresión múltiple",
        "ridge": f"Ridge (α={ridge_alpha:g})",
    }
    for key in ["simple", "multiple", "ridge"]:
        result = regression_result(df, key, alpha=ridge_alpha)
        rows.append({
            "Modelo": labels[key],
            "R² entrenamiento": result.r2_train,
            "R² prueba": result.r2_test,
            "RMSE": result.rmse,
            "MAE": result.mae,
        })
    return pd.DataFrame(rows)


def classification_result(
    df: pd.DataFrame,
    solver: str = "lbfgs",
    C: float = 1.0,
    threshold: float = 0.50,
) -> ClassificationResult:
    if solver not in {"lbfgs", "liblinear"}:
        raise ValueError("solver debe ser lbfgs o liblinear")

    base = df[["weight", "mpg"]].dropna().copy()
    y = (base["mpg"] > MPG_THRESHOLD).astype(int)
    X = base[["weight"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=float(C), solver=solver, max_iter=2000)),
    ])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= float(threshold)).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fpr, tpr, _ = roc_curve(y_test, prob)

    return ClassificationResult(
        solver=solver,
        C=float(C),
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_test, pred)),
        sensitivity=float(recall_score(y_test, pred, zero_division=0)),
        specificity=float(specificity),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        auc=float(roc_auc_score(y_test, prob)),
        cm=cm,
        y_test=y_test,
        probabilities=np.asarray(prob),
        predictions=np.asarray(pred),
        fpr=np.asarray(fpr),
        tpr=np.asarray(tpr),
        test_weight=X_test["weight"],
    )


def hyperparameter_grid(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for solver in ["liblinear", "lbfgs"]:
        for C in [0.01, 0.1, 1, 10, 100]:
            result = classification_result(df, solver=solver, C=C, threshold=threshold)
            rows.append({
                "solver": solver,
                "C": C,
                "accuracy": result.accuracy,
                "sensitivity": result.sensitivity,
                "specificity": result.specificity,
                "auc": result.auc,
            })
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["mpg", "weight", "displacement", "acceleration", "horsepower"]
    return df[cols].corr(numeric_only=True)
