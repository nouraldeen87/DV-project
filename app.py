import os
import glob
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# ─────────────────────────────────────────────────────────────
# 1) LOAD ONLY NEEDED COLUMNS + OPTIMIZE DTYPES
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "output_chunks"

# Only load columns the app actually uses
USECOLS = ["title", "rating", "pages", "totalratings", "bookformat", "genre"]

# Force memory-efficient dtypes on read
DTYPES = {
    "title":        "string",
    "bookformat":   "category",
    "genre":        "category",
    "rating":       "float32",    # 4 bytes vs 8 bytes
    "pages":        "float32",
    "totalratings": "float32",
}

part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "output_part_*.csv")))

if not part_files:
    raise FileNotFoundError(
        f"No CSV parts found in '{OUTPUT_DIR}/'. "
        "Make sure output_part_1.csv ... output_part_N.csv exist."
    )

print(f"📂 Loading {len(part_files)} file(s)...")

chunks = []
for f in part_files:
    chunk = pd.read_csv(f, usecols=USECOLS, dtype=DTYPES)
    chunks.append(chunk)
    print(f"   ✅ {os.path.basename(f)} — {chunk.memory_usage(deep=True).sum() / 1024**2:.1f} MB in RAM")

df = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory immediately after concat

# ─────────────────────────────────────────────────────────────
# 2) CLEANING
# ─────────────────────────────────────────────────────────────
df["title"] = df["title"].fillna("Unknown")
df = df.dropna(subset=["bookformat", "genre"]).copy()

df["genre"]      = df["genre"].astype(str).str.split(",").str[0].str.strip().astype("category")
df["bookformat"] = df["bookformat"].astype(str).str.strip().str.title().astype("category")

df["totalratings"] = pd.to_numeric(df["totalratings"], errors="coerce").astype("float32")
df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce").astype("float32")
df["pages"]        = pd.to_numeric(df["pages"],        errors="coerce").astype("float32")

df = df.dropna(subset=["rating", "pages", "totalratings"]).copy()
df = df[df["pages"] < 3000].copy()

# Pre-sample scatter data ONCE at startup — not on every callback
SCATTER_SAMPLE = df.sample(min(3000, len(df)), random_state=42).copy()
SCATTER_SAMPLE["log_tr"] = np.log1p(SCATTER_SAMPLE["totalratings"])

top_formats = df["bookformat"].value_counts().head(6).index.tolist()

mem_mb = df.memory_usage(deep=True).sum() / 1024**2
print(f"🎉 Dataset ready: {len(df):,} rows | RAM used: {mem_mb:.1f} MB")

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
    Output("genre-bar",     "figure"),
    Output("scatter-chart", "figure"),
    Output("kpi-row",       "children"),
    Input("format-dropdown", "value"),
    Input("rating-slider",   "value"),
)
def update(selected_formats, rating_range):

    if not selected_formats:
        selected_formats = top_formats[:3]

    low, high = rating_range

    # Filter main df for KPIs + bar chart
    mask = (
        df["bookformat"].isin(selected_formats) &
        (df["rating"] >= low) &
        (df["rating"] <= high)
    )
    filtered = df[mask]

    # Filter pre-sampled scatter data (no re-sampling every callback)
    scatter_mask = (
        SCATTER_SAMPLE["bookformat"].isin(selected_formats) &
        (SCATTER_SAMPLE["rating"] >= low) &
        (SCATTER_SAMPLE["rating"] <= high)
    )
    scat = SCATTER_SAMPLE[scatter_mask]

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

    avg_rating   = round(float(filtered["rating"].mean()), 2) if len(filtered) else 0
    median_pages = int(filtered["pages"].median())            if len(filtered) else 0

    kpis = [
        card("Books",        f"{len(filtered):,}"),
        card("Avg Rating",   avg_rating),
        card("Median Pages", median_pages),
    ]

    # ── Bar chart ────────────────────────────────────────
    gc = filtered["genre"].value_counts().head(10).reset_index()
    gc.columns = ["Genre", "Books"]
    bar = px.bar(gc, x="Books", y="Genre", orientation="h")

    # ── Scatter ──────────────────────────────────────────
    scatter = px.scatter(
        scat if len(scat) > 0 else SCATTER_SAMPLE.head(0),
        x="pages",
        y="rating",
        color="bookformat",
        size="log_tr",
        hover_data=["title"]
    ) if len(scat) > 0 else px.scatter()

    return bar, scatter, kpis

# ─────────────────────────────────────────────────────────────
# 5) RUN (LOCAL + RENDER SAFE)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
