import os
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from dash import Dash, dcc, html, Input, Output

# ─────────────────────────────────────────────────────────────
# 1) DATA AUTO-DOWNLOAD (FIX FOR GITHUB SIZE LIMITS)
# ─────────────────────────────────────────────────────────────
DATA_PATH = "train.parquet"

URL = "https://huggingface.co/datasets/euclaise/goodreads_100k/resolve/main/train.parquet"

if not os.path.exists(DATA_PATH):
    print("Downloading dataset...")
    r = requests.get(URL)
    r.raise_for_status()
    with open(DATA_PATH, "wb") as f:
        f.write(r.content)

df = pd.read_parquet(DATA_PATH)

# ─────────────────────────────────────────────────────────────
# 2) CLEANING
# ─────────────────────────────────────────────────────────────
df.columns = df.columns.str.lower()

df["title"] = df["title"].fillna("Unknown")
df = df.dropna(subset=["bookformat", "genre"]).copy()

df["genre"] = df["genre"].astype(str).str.split(",").str[0].str.strip()
df["bookformat"] = df["bookformat"].astype(str).str.strip().str.title()

df["totalratings"] = pd.to_numeric(df["totalratings"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["pages"] = pd.to_numeric(df["pages"], errors="coerce")

df = df.dropna(subset=["rating", "pages", "totalratings"]).copy()
df = df[df["pages"] < 3000].copy()

top_formats = df["bookformat"].value_counts().head(6).index.tolist()

# ─────────────────────────────────────────────────────────────
# 3) DASH APP
# ─────────────────────────────────────────────────────────────
app = Dash(__name__)
server = app.server  # REQUIRED FOR RENDER

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("📚 GoodReads Dashboard",
                style={"margin": "0", "color": "white"}),
        html.P("Interactive Analytics App",
               style={"margin": "4px 0 0", "color": "#cfe2ff"})
    ], style={"background": "#1a3c6b", "padding": "20px"}),

    # Controls
    html.Div([
        dcc.Dropdown(
            id="format-dropdown",
            options=[{"label": f, "value": f} for f in top_formats],
            value=top_formats[:3],
            multi=True
        ),
        dcc.RangeSlider(
            id="rating-slider",
            min=0, max=5, step=0.1,
            value=[3.0, 5.0],
            marks={i: str(i) for i in range(6)}
        )
    ], style={"padding": "20px"}),

    # KPIs
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "10px", "padding": "20px"}),

    # Charts
    html.Div([
        dcc.Graph(id="genre-bar"),
        dcc.Graph(id="scatter-chart")
    ], style={"display": "flex", "flexWrap": "wrap"}),

])

# ─────────────────────────────────────────────────────────────
# 4) CALLBACKS
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("genre-bar", "figure"),
    Output("scatter-chart", "figure"),
    Output("kpi-row", "children"),
    Input("format-dropdown", "value"),
    Input("rating-slider", "value"),
)
def update(selected_formats, rating_range):

    if not selected_formats:
        selected_formats = top_formats[:3]

    low, high = rating_range

    filtered = df[
        (df["bookformat"].isin(selected_formats)) &
        (df["rating"] >= low) &
        (df["rating"] <= high)
    ]

    # ── KPIs ─────────────────────────────────────────────
    def card(label, value):
        return html.Div([
            html.P(label),
            html.H3(value)
        ], style={
            "padding": "10px",
            "background": "white",
            "borderRadius": "8px",
            "flex": "1"
        })

    avg_rating = round(filtered["rating"].mean(), 2) if len(filtered) else 0
    median_pages = int(filtered["pages"].median()) if len(filtered) else 0

    kpis = [
        card("Books", len(filtered)),
        card("Avg Rating", avg_rating),
        card("Median Pages", median_pages),
    ]

    # ── Bar chart ───────────────────────────────────────
    gc = filtered["genre"].value_counts().head(10).reset_index()
    gc.columns = ["Genre", "Books"]
    bar = px.bar(gc, x="Books", y="Genre", orientation="h")

    # ── Scatter ─────────────────────────────────────────
    if len(filtered) > 0:
        scat = filtered.sample(min(3000, len(filtered)), random_state=42).copy()
        scat["log_tr"] = np.log1p(scat["totalratings"])

        scatter = px.scatter(
            scat,
            x="pages",
            y="rating",
            color="bookformat",
            size="log_tr",
            hover_data=["title"]
        )
    else:
        scatter = px.scatter()

    return bar, scatter, kpis


# ─────────────────────────────────────────────────────────────
# 5) RUN (LOCAL + RENDER SAFE)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
