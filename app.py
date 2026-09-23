from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dash_table, dcc, html

from model_utils import (
    MPG_THRESHOLD,
    ORIGIN_LABELS,
    add_labels,
    classification_result,
    correlation_matrix,
    filter_data,
    hyperparameter_grid,
    load_auto_mpg,
    model_comparison,
    pearson_weight_mpg,
    regression_result,
)

# Paleta deliberadamente sin azul, en continuidad con la presentación del informe.
WINE = "#7A1F3D"
WINE_DARK = "#54142A"
SAND = "#D4A373"
CHARCOAL = "#222222"
GRAY = "#6B6B6B"
LIGHT = "#F6F3EF"
WHITE = "#FFFFFF"
PLOT_COLORS = [WINE, SAND, "#8A6D5D", "#4F4F4F", "#A44A3F"]

DF = add_labels(load_auto_mpg(Path(__file__).resolve().parent / "data"))
YEAR_MIN = int(DF["model_year"].min())
YEAR_MAX = int(DF["model_year"].max())
ALL_ORIGINS = sorted(int(v) for v in DF["origin"].dropna().unique())
ALL_CYLINDERS = sorted(int(v) for v in DF["cylinders"].dropna().unique())

_hub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
_dash_kwargs = {}
if _hub_prefix:
    # Binder/JupyterHub expone el puerto 8050 a través de jupyter-server-proxy.
    _dash_kwargs["requests_pathname_prefix"] = f"{_hub_prefix.rstrip('/')}/proxy/8050/"

app = Dash(
    __name__,
    title="Auto MPG | Proyecto final",
    suppress_callback_exceptions=True,
    **_dash_kwargs,
)
server = app.server


def card(title: str, value_id: str, subtitle: str = "") -> html.Div:
    return html.Div(
        [html.Div(title, className="kpi-title"), html.Div(id=value_id, className="kpi-value"), html.Div(subtitle, className="kpi-subtitle")],
        className="kpi-card",
    )


def section_title(title: str, text: str | None = None) -> html.Div:
    children = [html.H2(title)]
    if text:
        children.append(html.P(text, className="section-intro"))
    return html.Div(children, className="section-heading")


GLOBAL_FILTERS = html.Div(
    [
        html.Div(
            [
                html.Label("Año del modelo"),
                dcc.RangeSlider(
                    YEAR_MIN,
                    YEAR_MAX,
                    1,
                    value=[YEAR_MIN, YEAR_MAX],
                    marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 2)},
                    id="year-filter",
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
            className="filter-block wide-filter",
        ),
        html.Div(
            [
                html.Label("Origen"),
                dcc.Dropdown(
                    options=[{"label": ORIGIN_LABELS.get(v, str(v)), "value": v} for v in ALL_ORIGINS],
                    value=ALL_ORIGINS,
                    multi=True,
                    id="origin-filter",
                    clearable=False,
                ),
            ],
            className="filter-block",
        ),
        html.Div(
            [
                html.Label("Cilindros"),
                dcc.Dropdown(
                    options=[{"label": str(v), "value": v} for v in ALL_CYLINDERS],
                    value=ALL_CYLINDERS,
                    multi=True,
                    id="cyl-filter",
                    clearable=False,
                ),
            ],
            className="filter-block",
        ),
    ],
    className="filter-panel",
)


app.layout = html.Div(
    [
        html.Header(
            [
                html.Div("PROYECTO FINAL · ETAPA DE TRANSFERENCIA", className="eyebrow"),
                html.H1("Auto MPG: eficiencia de combustible y modelado estadístico"),
                html.P(
                    "Dashboard interactivo para explorar la relación entre peso y MPG, comparar modelos "
                    "de regresión y evaluar una clasificación de eficiencia.",
                    className="hero-copy",
                ),
                html.Div(
                    [
                        html.Span("José Felipe Martínez Zafra"),
                        html.Span("UCompensar"),
                        html.Span("Septiembre de 2026"),
                    ],
                    className="meta-row",
                ),
            ],
            className="hero",
        ),
        html.Main(
            [
                GLOBAL_FILTERS,
                dcc.Tabs(
                    id="main-tabs",
                    value="tab-summary",
                    children=[
                        dcc.Tab(label="Resumen", value="tab-summary"),
                        dcc.Tab(label="Regresión", value="tab-regression"),
                        dcc.Tab(label="Clasificación", value="tab-classification"),
                        dcc.Tab(label="Datos y trazabilidad", value="tab-data"),
                    ],
                ),
                html.Div(id="tab-content"),
            ],
            className="page-shell",
        ),
        html.Footer(
            "Auto MPG · Dash + Plotly + Pandas + SciPy + scikit-learn · random_state = 42",
            className="footer",
        ),
    ]
)


def summary_layout() -> html.Div:
    return html.Div(
        [
            section_title(
                "Resumen del problema",
                "Los filtros modifican la exploración descriptiva. Las métricas base del proyecto se calculan "
                "sobre las 398 observaciones para conservar comparabilidad con las etapas anteriores.",
            ),
            html.Div(
                [
                    card("Observaciones filtradas", "kpi-n", "Registros visibles"),
                    card("Correlación peso–MPG", "kpi-r", "Pearson unilateral"),
                    card("MPG mediano", "kpi-median", "Umbral de clasificación"),
                    card("MPG medio", "kpi-mean", "Según filtros activos"),
                ],
                className="kpi-grid",
            ),
            html.Div(
                [dcc.Graph(id="scatter-weight-mpg", className="chart-card"), dcc.Graph(id="hist-mpg", className="chart-card")],
                className="two-col",
            ),
            html.Div(
                [dcc.Graph(id="box-origin", className="chart-card"), html.Div(id="summary-story", className="narrative-card")],
                className="two-col",
            ),
        ],
        className="tab-panel",
    )


def regression_layout() -> html.Div:
    return html.Div(
        [
            section_title(
                "Comparación de modelos de regresión",
                "La pregunta no es cuál modelo tiene más variables, sino cuál aporta mejor generalización y una "
                "interpretación defendible.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Modelo"),
                            dcc.Dropdown(
                                id="reg-model",
                                value="simple",
                                clearable=False,
                                options=[
                                    {"label": "Regresión lineal simple", "value": "simple"},
                                    {"label": "Regresión lineal múltiple", "value": "multiple"},
                                    {"label": "Ridge", "value": "ridge"},
                                ],
                            ),
                        ],
                        className="filter-block",
                    ),
                    html.Div(
                        [
                            html.Label("Alpha de Ridge"),
                            dcc.Slider(
                                id="ridge-alpha",
                                min=-2,
                                max=2,
                                step=0.25,
                                value=0,
                                marks={-2: "0.01", -1: "0.1", 0: "1", 1: "10", 2: "100"},
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        className="filter-block wide-filter",
                    ),
                ],
                className="control-row",
            ),
            html.Div(
                [card("R² prueba", "reg-r2"), card("RMSE", "reg-rmse", "MPG"), card("MAE", "reg-mae", "MPG"), card("R² entrenamiento", "reg-r2-train")],
                className="kpi-grid",
            ),
            html.Div(
                [dcc.Graph(id="reg-pred-vs-real", className="chart-card"), dcc.Graph(id="reg-residuals", className="chart-card")],
                className="two-col",
            ),
            html.Div(
                [dcc.Graph(id="reg-coefficients", className="chart-card"), dcc.Graph(id="reg-comparison", className="chart-card")],
                className="two-col",
            ),
            html.Div(id="reg-interpretation", className="narrative-card"),
        ],
        className="tab-panel",
    )


def classification_layout() -> html.Div:
    return html.Div(
        [
            section_title(
                "Clasificación de eficiencia",
                f"La clase alta se define como MPG > {MPG_THRESHOLD:.1f}. Ajusta C, solver y el umbral de decisión para observar el compromiso entre errores.",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("Solver"), dcc.Dropdown(id="clf-solver", value="lbfgs", clearable=False, options=[{"label": "lbfgs", "value": "lbfgs"}, {"label": "liblinear", "value": "liblinear"}])],
                        className="filter-block",
                    ),
                    html.Div(
                        [html.Label("C (inverso de regularización)"), dcc.Dropdown(id="clf-c", value=1.0, clearable=False, options=[{"label": str(v), "value": float(v)} for v in [0.01, 0.1, 1, 10, 100]])],
                        className="filter-block",
                    ),
                    html.Div(
                        [html.Label("Umbral de clasificación"), dcc.Slider(id="clf-threshold", min=0.2, max=0.8, step=0.05, value=0.5, marks={0.2: "0.2", 0.5: "0.5", 0.8: "0.8"}, tooltip={"placement": "bottom", "always_visible": False})],
                        className="filter-block wide-filter",
                    ),
                ],
                className="control-row",
            ),
            html.Div(
                [card("Exactitud", "clf-acc"), card("Sensibilidad", "clf-sens"), card("Especificidad", "clf-spec"), card("AUC", "clf-auc")],
                className="kpi-grid",
            ),
            html.Div(
                [dcc.Graph(id="clf-confusion", className="chart-card"), dcc.Graph(id="clf-roc", className="chart-card")],
                className="two-col",
            ),
            html.Div(
                [dcc.Graph(id="clf-probability", className="chart-card"), dcc.Graph(id="clf-grid", className="chart-card")],
                className="two-col",
            ),
            html.Div(id="clf-interpretation", className="narrative-card"),
        ],
        className="tab-panel",
    )


def data_layout() -> html.Div:
    return html.Div(
        [
            section_title(
                "Datos, correlaciones y trazabilidad",
                "La tabla responde a los filtros globales. La matriz de correlación se calcula sobre las variables numéricas principales del subconjunto visible.",
            ),
            dcc.Graph(id="corr-heatmap", className="chart-card"),
            html.Div(id="data-count-note", className="mini-note"),
            dash_table.DataTable(
                id="data-table",
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={"fontWeight": "700", "backgroundColor": LIGHT, "color": CHARCOAL},
                style_cell={"fontFamily": "Arial, sans-serif", "fontSize": 13, "padding": "8px", "textAlign": "left", "maxWidth": 180, "whiteSpace": "normal"},
            ),
            html.Div(
                [
                    html.H3("Decisiones metodológicas"),
                    html.Ul(
                        [
                            html.Li("Partición 70/30 con random_state = 42 para regresión."),
                            html.Li("Partición estratificada 70/30 para clasificación."),
                            html.Li("StandardScaler dentro de Pipeline para evitar fuga de datos."),
                            html.Li("Comparación explícita entre modelo simple, múltiple y Ridge."),
                            html.Li("Exploración de C, solver y umbral en regresión logística."),
                            html.Li("Las asociaciones estadísticas no se interpretan como causalidad."),
                        ]
                    ),
                ],
                className="narrative-card",
            ),
        ],
        className="tab-panel",
    )


@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    return {
        "tab-summary": summary_layout(),
        "tab-regression": regression_layout(),
        "tab-classification": classification_layout(),
        "tab-data": data_layout(),
    }.get(tab, summary_layout())


def _filtered(year_range, origins, cylinders) -> pd.DataFrame:
    return filter_data(DF, year_range=year_range, origins=origins, cylinders=cylinders)


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16, "color": GRAY})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=380)
    return fig


@callback(
    Output("kpi-n", "children"),
    Output("kpi-r", "children"),
    Output("kpi-median", "children"),
    Output("kpi-mean", "children"),
    Output("scatter-weight-mpg", "figure"),
    Output("hist-mpg", "figure"),
    Output("box-origin", "figure"),
    Output("summary-story", "children"),
    Input("year-filter", "value"),
    Input("origin-filter", "value"),
    Input("cyl-filter", "value"),
    prevent_initial_call=False,
)
def update_summary(year_range, origins, cylinders):
    sub = _filtered(year_range, origins, cylinders)
    if len(sub) < 3:
        empty = _empty_figure("No hay suficientes datos con estos filtros.")
        return str(len(sub)), "—", f"{MPG_THRESHOLD:.1f}", "—", empty, empty, empty, html.P("Amplía los filtros para continuar la exploración.")

    r, p = pearson_weight_mpg(sub)
    scatter = px.scatter(
        sub,
        x="weight",
        y="mpg",
        color="origin_label",
        hover_name="name",
        labels={"weight": "Peso (lb)", "mpg": "Millas por galón (MPG)", "origin_label": "Origen"},
        title="Peso frente a eficiencia de combustible",
        color_discrete_sequence=PLOT_COLORS,
    )
    x = sub["weight"].to_numpy(float)
    y = sub["mpg"].to_numpy(float)
    if len(np.unique(x)) > 1:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        scatter.add_trace(go.Scatter(x=xx, y=m * xx + b, mode="lines", name="Tendencia lineal", line={"color": CHARCOAL, "width": 2}))
    scatter.update_layout(template="plotly_white", legend_title_text="Origen")

    hist = px.histogram(sub, x="mpg", nbins=14, title="Distribución de MPG", labels={"mpg": "MPG", "count": "Frecuencia"}, color_discrete_sequence=[WINE])
    hist.add_vline(x=MPG_THRESHOLD, line_dash="dash", line_color=CHARCOAL, annotation_text="Umbral 23 MPG")
    hist.update_layout(template="plotly_white", showlegend=False)

    box = px.box(sub, x="origin_label", y="mpg", points="outliers", title="MPG por origen", labels={"origin_label": "Origen", "mpg": "MPG"}, color="origin_label", color_discrete_sequence=PLOT_COLORS)
    box.update_layout(template="plotly_white", showlegend=False)

    direction = "negativa" if r < 0 else "positiva"
    story = [
        html.H3("Lectura del subconjunto"),
        html.P(f"Con los filtros activos quedan {len(sub)} vehículos. La correlación peso–MPG es {direction} (r = {r:.3f})."),
        html.P(f"El p-valor unilateral es {p:.2e}. En el proyecto completo, la evidencia sustenta una relación negativa fuerte, pero no una relación causal."),
        html.P("La visualización permite comprobar si el patrón general se mantiene o cambia al segmentar por año, origen y número de cilindros."),
    ]
    return str(len(sub)), f"{r:.3f}", f"{MPG_THRESHOLD:.1f}", f"{sub['mpg'].mean():.2f}", scatter, hist, box, story


@callback(
    Output("reg-r2", "children"),
    Output("reg-rmse", "children"),
    Output("reg-mae", "children"),
    Output("reg-r2-train", "children"),
    Output("reg-pred-vs-real", "figure"),
    Output("reg-residuals", "figure"),
    Output("reg-coefficients", "figure"),
    Output("reg-comparison", "figure"),
    Output("reg-interpretation", "children"),
    Input("reg-model", "value"),
    Input("ridge-alpha", "value"),
)
def update_regression(model_name, alpha_exp):
    alpha = 10 ** float(alpha_exp)
    result = regression_result(DF, model_name=model_name, alpha=alpha)

    pred_df = pd.DataFrame({"Real": result.y_test.to_numpy(), "Predicho": result.predictions})
    pred_fig = px.scatter(pred_df, x="Real", y="Predicho", title="MPG real frente a MPG predicho", color_discrete_sequence=[WINE])
    lo = min(pred_df.min())
    hi = max(pred_df.max())
    pred_fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Predicción perfecta", line={"color": CHARCOAL, "dash": "dash"}))
    pred_fig.update_layout(template="plotly_white")

    resid_df = pd.DataFrame({"Predicho": result.predictions, "Residuo": result.residuals})
    resid_fig = px.scatter(resid_df, x="Predicho", y="Residuo", title="Residuos frente a valores predichos", color_discrete_sequence=[SAND])
    resid_fig.add_hline(y=0, line_dash="dash", line_color=CHARCOAL)
    resid_fig.update_layout(template="plotly_white")

    coef_df = pd.DataFrame({"Variable": list(result.coefficients.keys()), "Coeficiente": list(result.coefficients.values())})
    coef_fig = px.bar(coef_df, x="Variable", y="Coeficiente", title="Coeficientes del modelo", color_discrete_sequence=[WINE])
    coef_fig.update_layout(template="plotly_white", showlegend=False)

    comp = model_comparison(DF, ridge_alpha=alpha)
    comp_fig = px.bar(comp, x="Modelo", y="R² prueba", title="Comparación de R² en prueba", text_auto=".3f", color="Modelo", color_discrete_sequence=PLOT_COLORS)
    comp_fig.update_layout(template="plotly_white", showlegend=False, yaxis_range=[0, max(0.8, comp["R² prueba"].max() + 0.05)])

    labels = {"simple": "regresión simple", "multiple": "regresión múltiple", "ridge": "Ridge"}
    text = [html.H3("Interpretación"), html.P(f"El modelo seleccionado es {labels[model_name]}. Su R² en prueba es {result.r2_test:.4f} y el RMSE es {result.rmse:.4f} MPG.")]
    if model_name == "simple":
        text.append(html.P("En la partición documentada, este modelo supera ligeramente a la versión múltiple. Es la referencia predictiva más parsimoniosa."))
    elif model_name == "multiple":
        text.append(html.P("El modelo múltiple aporta interpretación conjunta, pero weight y displacement son altamente correlacionados, por lo que los coeficientes individuales deben leerse con cautela."))
    else:
        text.append(html.P(f"Ridge usa α = {alpha:g}. La regularización es metodológicamente pertinente ante multicolinealidad, aunque α = 1 no mejoró el desempeño base en la evidencia del proyecto."))

    return f"{result.r2_test:.4f}", f"{result.rmse:.4f}", f"{result.mae:.4f}", f"{result.r2_train:.4f}", pred_fig, resid_fig, coef_fig, comp_fig, text


@callback(
    Output("clf-acc", "children"),
    Output("clf-sens", "children"),
    Output("clf-spec", "children"),
    Output("clf-auc", "children"),
    Output("clf-confusion", "figure"),
    Output("clf-roc", "figure"),
    Output("clf-probability", "figure"),
    Output("clf-grid", "figure"),
    Output("clf-interpretation", "children"),
    Input("clf-solver", "value"),
    Input("clf-c", "value"),
    Input("clf-threshold", "value"),
)
def update_classification(solver, C, threshold):
    result = classification_result(DF, solver=solver, C=float(C), threshold=float(threshold))

    cm_fig = go.Figure(data=go.Heatmap(z=result.cm, x=["Pred. baja", "Pred. alta"], y=["Real baja", "Real alta"], text=result.cm, texttemplate="%{text}", colorscale=[[0, LIGHT], [1, WINE]], showscale=False))
    cm_fig.update_layout(title="Matriz de confusión", template="plotly_white", yaxis_autorange="reversed")

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=result.fpr, y=result.tpr, mode="lines", name=f"ROC (AUC={result.auc:.3f})", line={"color": WINE, "width": 3}))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar", line={"color": CHARCOAL, "dash": "dash"}))
    roc_fig.update_layout(title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad", template="plotly_white")

    prob_df = pd.DataFrame({"weight": result.test_weight.to_numpy(), "prob": result.probabilities, "real": np.where(result.y_test.to_numpy() == 1, "Alta", "Baja/mediana")})
    prob_fig = px.scatter(prob_df, x="weight", y="prob", color="real", title="Probabilidad estimada de eficiencia alta", labels={"weight": "Peso (lb)", "prob": "P(MPG > 23)", "real": "Clase real"}, color_discrete_sequence=PLOT_COLORS)
    prob_fig.add_hline(y=float(threshold), line_dash="dash", line_color=CHARCOAL, annotation_text=f"Umbral {threshold:.2f}")
    prob_fig.update_layout(template="plotly_white")

    grid = hyperparameter_grid(DF, threshold=0.5)
    grid_fig = px.line(grid, x="C", y="accuracy", color="solver", markers=True, log_x=True, title="Exactitud según C y solver (umbral 0.50)", labels={"accuracy": "Exactitud", "C": "C", "solver": "Solver"}, color_discrete_sequence=[WINE, SAND])
    grid_fig.update_layout(template="plotly_white", yaxis_range=[0.75, 0.9])

    interpretation = [
        html.H3("Interpretación"),
        html.P(f"Con solver={solver}, C={float(C):g} y umbral={float(threshold):.2f}, la exactitud es {result.accuracy:.2%}, la sensibilidad {result.sensitivity:.2%} y la especificidad {result.specificity:.2%}."),
        html.P(f"El AUC = {result.auc:.4f} resume la capacidad discriminativa para todos los umbrales. Cambiar el umbral altera la matriz de confusión, pero no el AUC."),
        html.P("La configuración documentada del proyecto es lbfgs, C = 1 y umbral = 0.50."),
    ]

    return f"{result.accuracy:.2%}", f"{result.sensitivity:.2%}", f"{result.specificity:.2%}", f"{result.auc:.4f}", cm_fig, roc_fig, prob_fig, grid_fig, interpretation


@callback(
    Output("corr-heatmap", "figure"),
    Output("data-table", "data"),
    Output("data-table", "columns"),
    Output("data-count-note", "children"),
    Input("year-filter", "value"),
    Input("origin-filter", "value"),
    Input("cyl-filter", "value"),
)
def update_data(year_range, origins, cylinders):
    sub = _filtered(year_range, origins, cylinders)
    if len(sub) < 3:
        fig = _empty_figure("No hay suficientes datos para calcular correlaciones.")
    else:
        corr = correlation_matrix(sub)
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=[[0, WINE], [0.5, WHITE], [1, SAND]], text=np.round(corr.values, 2), texttemplate="%{text}", colorbar={"title": "r"}))
        fig.update_layout(title="Matriz de correlaciones", template="plotly_white")

    cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin_label", "name"]
    table_df = sub[cols].copy().round(3)
    columns = [{"name": c.replace("_", " ").title(), "id": c} for c in table_df.columns]
    return fig, table_df.to_dict("records"), columns, f"{len(table_df)} registros visibles de {len(DF)} observaciones totales."


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
