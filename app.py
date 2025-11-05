import os
import pickle
import re
from datetime import datetime

import pandas as pd
import plotly.express as px

# Dash modern imports, including Output and Input for callbacks
from dash import Dash, Input, Output, dash_table, dcc, html
from flask import Flask, render_template, request

# ============================================================
# Flask app (prediksi + util routes)
# ============================================================
app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# --- Load model(s) ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} tidak ditemukan. Pastikan file ada.")

with open(MODEL_PATH, "rb") as f:
    loaded = pickle.load(f)

models = list(loaded) if isinstance(loaded, (list, tuple)) else [loaded]


def pretty_name(estimator):
    cls = estimator.__class__.__name__
    if cls == "DecisionTreeClassifier":
        return "Decision Tree"
    if cls == "SVC":
        return "SVC"
    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", cls)


model_display_names = [pretty_name(m) for m in models]

# --- Load scaler ---
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"{SCALER_PATH} tidak ditemukan. Pastikan file ada.")

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# --- Determine expected feature names ---
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    feature_names = ["lifeExp", "gdpPercap"]


# ---------- Example Flask utility routes ----------
@app.route("/hello/<name>")
def hello_there(name):
    now = datetime.now()
    formatted_now = now.strftime("%A, %d %B, %Y at %X")
    match_object = re.match(r"[a-zA-Z]+", name)
    clean_name = match_object.group(0) if match_object else "Friend"
    return f"Hello there, {clean_name}! It's {formatted_now}"


@app.route("/pyramid/<height>")
def pyramid(height):
    height = int(height)
    out = []
    for i in range(height):
        out.append(" " * (height - i - 1) + "* " * (2 * i + 1) + "<br>")
    return "".join(out)


# ---------- Prediction UI ----------
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_display_names=model_display_names,
        feature_names=feature_names,
    )


@app.route("/predict", methods=["POST"])
def predict():
    raw_idx = request.form.get("model")
    try:
        sel_idx = int(raw_idx)
    except Exception:
        sel_idx = 0
    if not (0 <= sel_idx < len(models)):
        sel_idx = 0

    input_values = {}
    for feat in feature_names:
        raw = request.form.get(feat, "").strip()
        if raw == "":
            return render_template(
                "index.html",
                model_display_names=model_display_names,
                feature_names=feature_names,
                prediction=f"Missing input for '{feat}'.",
            )
        try:
            input_values[feat] = float(raw)
        except ValueError:
            return render_template(
                "index.html",
                model_display_names=model_display_names,
                feature_names=feature_names,
                prediction=f"Input untuk '{feat}' harus angka.",
            )

    X_df = pd.DataFrame([input_values], columns=feature_names)

    try:
        X_scaled = scaler.transform(X_df)
    except Exception as e:
        return render_template(
            "index.html",
            model_display_names=model_display_names,
            feature_names=feature_names,
            prediction=f"Error saat mengaplikasikan scaler: {e}",
        )

    model_chosen = models[sel_idx]
    try:
        y_pred = model_chosen.predict(X_scaled)
        result = y_pred[0] if hasattr(y_pred, "__iter__") else y_pred
    except Exception as e:
        return render_template(
            "index.html",
            model_display_names=model_display_names,
            feature_names=feature_names,
            prediction=f"Error saat prediksi: {e}",
        )

    used_model = model_display_names[sel_idx]
    return render_template(
        "index.html",
        model_display_names=model_display_names,
        feature_names=feature_names,
        prediction=f"Model: {used_model} → Prediction: {result}",
        input_values=input_values,
    )


# ============================================================
# Dash App: Controls -> /dash/controls/
# ============================================================
def register_dash_controls(server: Flask) -> Dash:
    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    dash_app = Dash(
        __name__,
        server=server,
        external_stylesheets=external_stylesheets,
        routes_pathname_prefix="/dash/controls/",
        suppress_callback_exceptions=True,
    )

    df = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv"
    )

    dash_app.layout = html.Div(
        [
            html.Div(
                className="row",
                children="My First App with Data, Graph, and Controls",
                style={"textAlign": "center", "color": "blue", "fontSize": 30},
            ),
            html.Div(
                className="row",
                children=[
                    dcc.RadioItems(
                        options=[
                            {"label": "Population (pop)", "value": "pop"},
                            {"label": "Life Expectancy (lifeExp)", "value": "lifeExp"},
                            {
                                "label": "GDP per capita (gdpPercap)",
                                "value": "gdpPercap",
                            },
                        ],
                        value="lifeExp",
                        inline=True,
                        id="my-radio-buttons-final",
                    )
                ],
            ),
            html.Div(
                className="row",
                children=[
                    html.Div(
                        className="six columns",
                        children=[
                            dash_table.DataTable(
                                data=df.to_dict("records"),
                                page_size=11,
                                style_table={"overflowX": "auto"},
                            )
                        ],
                    ),
                    html.Div(
                        className="six columns",
                        children=[dcc.Graph(id="histo-chart-final", figure={})],
                    ),
                ],
            ),
        ]
    )

    # Correct usage of Output and Input
    @dash_app.callback(
        Output("histo-chart-final", "figure"),
        Input("my-radio-buttons-final", "value"),
    )
    def update_graph(col_chosen):
        fig = px.histogram(
            df,
            x="continent",
            y=col_chosen,
            histfunc="avg",
            title=f"Average {col_chosen} by Continent",
        )
        return fig

    return dash_app


# ============================================================
# Dash App: Unfiltered -> /dash/unfiltered/
# ============================================================
def register_dash_unfiltered(server: Flask) -> Dash:
    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix="/dash/unfiltered/",
        suppress_callback_exceptions=True,
    )

    df = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv"
    )

    country_options = [{"label": c, "value": c} for c in sorted(df["country"].unique())]

    dash_app.layout = html.Div(
        [
            html.H1(children="Title of Dash App", style={"textAlign": "center"}),
            dcc.Dropdown(
                options=country_options, value="Canada", id="dropdown-selection"
            ),
            dcc.Graph(id="graph-content"),
        ]
    )

    @dash_app.callback(
        Output("graph-content", "figure"), Input("dropdown-selection", "value")
    )
    def update_graph(value):
        dff = df[df.country == value]
        return px.line(dff, x="year", y="pop", title=f"Population in {value} by Year")

    return dash_app


# Register Dash apps
dash_controls = register_dash_controls(app)
dash_unfiltered = register_dash_unfiltered(app)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
