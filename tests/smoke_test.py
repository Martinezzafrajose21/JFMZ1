"""Pruebas rápidas de estructura sin depender de la descarga de datos."""
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]

required = [
    ROOT / "app.py",
    ROOT / "model_utils.py",
    ROOT / "requirements.txt",
    ROOT / "assets" / "style.css",
    ROOT / "Dashboard_Auto_MPG.ipynb",
]

for path in required:
    assert path.exists(), f"Falta archivo: {path}"

py_compile.compile(str(ROOT / "app.py"), doraise=True)
py_compile.compile(str(ROOT / "model_utils.py"), doraise=True)

print("Smoke test estructural OK")
