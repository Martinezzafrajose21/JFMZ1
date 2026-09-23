"""Comprueba el proyecto con los datos incluidos, sin acceso a Internet."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    ALL_CYLINDERS,
    ALL_ORIGINS,
    DF,
    YEAR_MAX,
    YEAR_MIN,
    app,
    render_tab,
    update_classification,
    update_data,
    update_regression,
    update_summary,
)
from model_utils import pearson_weight_mpg  # noqa: E402

assert (ROOT / "data" / "auto-mpg.data").exists()
assert DF.shape[0] == 398
assert DF.horsepower.isna().sum() == 6
r, _ = pearson_weight_mpg(DF)
assert abs(r + 0.831741) < 0.00001

client = app.server.test_client()
assert client.get("/").status_code == 200
assert client.get("/_dash-layout").status_code == 200
assert client.get("/_dash-dependencies").status_code == 200
for tab in ("tab-summary", "tab-regression", "tab-classification", "tab-data"):
    assert render_tab(tab) is not None

summary = update_summary([YEAR_MIN, YEAR_MAX], ALL_ORIGINS, ALL_CYLINDERS)
assert summary[0] == "398" and summary[1] == "-0.832"
empty = update_summary([70, 70], [3], [8])
assert empty[0] == "0"
for model in ("simple", "multiple", "ridge"):
    result = update_regression(model, 0)
    assert len(result) == 9 and float(result[0]) > 0.7
for solver in ("lbfgs", "liblinear"):
    result = update_classification(solver, 1, 0.5)
    assert result[0] == "86.67%" and len(result) == 9
assert "398 registros" in update_data([YEAR_MIN, YEAR_MAX], ALL_ORIGINS, ALL_CYLINDERS)[-1]

print("Integración local OK: datos, HTTP, pestañas, filtros y modelos")
