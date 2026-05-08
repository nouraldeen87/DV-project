import os
import glob
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

# ─────────────────────────────────────────────────────────────
# 1) LOAD ALL CSV PARTS
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "output_chunks"

part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "output_part_*.csv")))

if not part_files:
    raise FileNotFoundError(
        f"No CSV parts found in '{OUTPUT_DIR}/'. "
        "Make sure output_part_1.csv ... output_part_N.csv exist."
    )

print(f"📂 Loading {len(part_files)} file(s):")
for f in part_files:
    size = os.path.getsize(f) / (1024 * 1024)
    print(f"   📦 {os.path.basename(f)} — {size:.2f} MB")

df = pd.concat([pd.read_csv(f) for f in part_files], ignore_index=True)
print(f"✅ Loaded {len(df):,} rows from all parts")

# ─────────────────────────────────────────────────────────────
# 2) CLEANING (mirrors Phase 1 / notebook pipeline)
# ─────────────────────────────────────────────────────────────
df.columns = df.columns.str.lower()

df["title"] = df["title"].fillna("Unknown")
df = df.dropna(subset=["bookformat", "genre"]).copy()

df["genre"]      = df["genre"].astype(str).str.split(",").str[0].str.strip()
df["bookformat"] = df["bookformat"].astype(str).str.strip().str.title()

df["totalratings"] = pd.to_numeric(df["totalratings"], errors="coerce")
df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
df["pages"]        = pd.to_numeric(df["pages"],        errors="coerce")
df["reviews"]      = pd.to_numeric(df.get("reviews"),  errors="coerce") if "reviews" in df.columns else 0

df = df.dropna(subset=["rating", "pages", "totalratings"]).copy()
df = df[df["pages"] < 3000].copy()

top_formats  = df["bookformat"].value_counts().head(6).index.tolist()
top3_formats = df["bookformat"].value_counts().head(3).index.tolist()
top4_formats = df["bookformat"].value_counts().head(4).index.tolist()
top8_genres  = df["genre"].value_counts().head(8).index.tolist()

print(f"🎉 Dataset ready: {len(df):,} rows | {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# 3) PREBUILT NOTEBOOK FIGURES (Task 1 — all 5)
# ─────────────────────────────────────────────────────────────

# ── Fig 1: Rating Distribution — Interactive Histogram ───────
mean_r   = df["rating"].mean()
median_r = df["rating"].median()

fig1 = px.histogram(
    df, x="rating", nbins=40,
    title="Distribution of Book Ratings — GoodReads 100k",
    labels={"rating": "Rating (0–5)", "count": "Number of Books"},
    color_discrete_sequence=["#4C72B0"],
    opacity=0.85,
    marginal="violin",
)
fig1.add_vline(x=mean_r,   line_color="crimson", line_dash="dash",
               annotation_text=f"Mean {mean_r:.2f}",   annotation_position="top right")
fig1.add_vline(x=median_r, line_color="green",   line_dash="dot",
               annotation_text=f"Median {median_r:.2f}", annotation_position="top left")
fig1.update_layout(
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    font=dict(family="Arial", size=13), title_font_size=16, bargap=0.05,
    xaxis=dict(title="Rating (0–5)", showgrid=True, gridcolor="#e0e0e0"),
    yaxis=dict(title="Number of Books", showgrid=True, gridcolor="#e0e0e0"),
    annotations=[dict(
        x=0.02, y=0.97, xref="paper", yref="paper",
        text="Ratings are left-skewed — most books cluster 3.5–4.2",
        showarrow=False, bgcolor="lightyellow", bordercolor="gray",
        font=dict(size=11, color="navy"),
    )],
)

# ── Fig 2: Top 10 Book Formats — Horizontal Bar ───────────────
top10_formats = df["bookformat"].value_counts().head(10).reset_index()
top10_formats.columns = ["bookformat", "count"]
top10_formats["share"]      = (top10_formats["count"] / len(df) * 100).round(1)
top10_formats["avg_rating"] = top10_formats["bookformat"].map(
    df.groupby("bookformat")["rating"].mean().round(2)
)

fig2 = px.bar(
    top10_formats, x="count", y="bookformat", orientation="h",
    color="count", color_continuous_scale="Blues",
    title="Top 10 Book Formats by Count",
    labels={"count": "Number of Books", "bookformat": ""},
    text="count",
    custom_data=["share", "avg_rating"],
    height=420,
)
fig2.update_traces(
    texttemplate="%{x:,}",
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Books: %{x:,}<br>Share: %{customdata[0]:.1f}%<br>Avg Rating: %{customdata[1]:.2f}<extra></extra>",
)
fig2.update_layout(
    yaxis=dict(autorange="reversed"),
    coloraxis_showscale=False,
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    xaxis=dict(title="Number of Books", showgrid=True, gridcolor="#ddd"),
    yaxis_title="",
    title_font_size=16, height=420,
)

# ── Fig 3: Pages vs Rating — Bubble Scatter ───────────────────
scat3 = df[df["bookformat"].isin(top_formats)].sample(
    min(5000, len(df)), random_state=42
).copy()
scat3["log_tr"] = np.log1p(scat3["totalratings"])

fig3 = px.scatter(
    scat3, x="pages", y="rating",
    color="bookformat", size="log_tr", size_max=16,
    opacity=0.55,
    hover_data={"title": True, "author": True, "pages": True,
                "rating": ":.2f", "totalratings": True, "log_tr": False},
    labels={"pages": "Pages", "rating": "Rating", "bookformat": "Format",
            "totalratings": "Total Ratings"},
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Pages vs. Rating — Bubble Size = log(Total Ratings)",
)
fig3.update_layout(
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    title_font_size=16, legend_title="Format",
    xaxis=dict(title="Pages", showgrid=True, gridcolor="#ddd"),
    yaxis=dict(title="Rating (0–5)", showgrid=True, gridcolor="#ddd"),
)

# ── Fig 4: Rating by Format — Multi-trace GO Histogram ────────
colors4 = ["#4C72B0", "#DD8452", "#55A868"]
fig4 = go.Figure()
for fmt, color in zip(top3_formats, colors4):
    subset = df[df["bookformat"] == fmt]["rating"]
    fig4.add_trace(go.Histogram(
        x=subset, name=fmt, nbinsx=35, opacity=0.65,
        marker_color=color,
        hovertemplate=f"<b>{fmt}</b><br>Rating: %{{x:.2f}}<br>Books: %{{y}}<extra></extra>",
    ))
for fmt, color in zip(top3_formats, colors4):
    mean_val = df[df["bookformat"] == fmt]["rating"].mean()
    fig4.add_vline(
        x=mean_val, line_color=color, line_dash="dash", line_width=2,
        annotation_text=f"{fmt} mean {mean_val:.2f}",
        annotation_font_color=color, annotation_font_size=10,
    )
fig4.update_layout(
    barmode="overlay",
    title="Rating Distribution: Paperback vs Hardcover vs Ebook (Multi-Trace)",
    title_font_size=16,
    xaxis=dict(title="Rating (0–5)", showgrid=True, gridcolor="#ddd"),
    yaxis=dict(title="Number of Books", showgrid=True, gridcolor="#ddd"),
    plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    legend=dict(title="Format", x=0.02, y=0.97,
                bgcolor="rgba(255,255,255,0.8)", bordercolor="gray"),
    annotations=[dict(
        x=0.5, y=-0.15, xref="paper", yref="paper",
        text="Click legend items to show/hide individual traces",
        showarrow=False, font=dict(size=11, color="gray"),
    )],
    height=480,
)

# ── Fig 5: Top Genres Animated Bar Chart ─────────────────────
df_anim = df[
    df["genre"].isin(top8_genres) & df["bookformat"].isin(top4_formats)
].groupby(["bookformat", "genre"], as_index=False).agg(
    count=("rating", "count"),
    avg_rating=("rating", "mean"),
).round({"avg_rating": 3})

genre_order = (
    df_anim.groupby("genre")["count"].sum()
    .sort_values(ascending=False).index.tolist()
)

fig5 = px.bar(
    df_anim, x="genre", y="count",
    animation_frame="bookformat",
    color="genre",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    title="Top 8 Genres by Book Count — Animated by Format",
    labels={"genre": "Genre", "count": "Number of Books",
            "avg_rating": "Avg Rating", "bookformat": "Format"},
    hover_data={"count": True, "avg_rating": ":.3f", "bookformat": False},
    category_orders={"genre": genre_order},
    text="count",
)
fig5.update_traces(
    texttemplate="%{y:,}",
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Books: %{y:,}<br>Avg Rating: %{customdata[0]:.2f}<extra></extra>",
)
fig5.update_layout(
    title_font_size=16, plot_bgcolor="#f9f9f9", paper_bgcolor="white",
    xaxis=dict(title="Genre", showgrid=False, tickangle=-20),
    yaxis=dict(title="Number of Books", showgrid=True, gridcolor="#ddd"),
    showlegend=False, height=500,
    annotations=[dict(
        x=0.5, y=-0.18, xref="paper", yref="paper",
        text="▶ Press Play to animate — each frame shows genre counts for one book format",
        showarrow=False, font=dict(size=11, color="gray"),
    )],
)
if fig5.layout.updatemenus:
    fig5.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
    fig5.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400

print("✅ All 5 notebook figures built")

# ─────────────────────────────────────────────────────────────
# 4) DASH APP
# ─────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # REQUIRED FOR RENDER

# ── Shared styles ─────────────────────────────────────────────
TAB_STYLE       = {"fontFamily": "Arial", "fontSize": "14px", "padding": "10px 20px"}
TAB_SEL_STYLE   = {**TAB_STYLE, "borderTop": "3px solid #1a3c6b",
                   "fontWeight": "bold", "color": "#1a3c6b"}
CARD_STYLE      = {
    "background": "white", "borderRadius": "8px",
    "padding": "12px 20px", "boxShadow": "0 1px 6px rgba(0,0,0,0.1)",
    "minWidth": "130px", "flex": "1",
}
CHART_CARD_STYLE = {
    "background": "white", "borderRadius": "8px", "padding": "16px",
    "boxShadow": "0 1px 6px rgba(0,0,0,0.1)", "flex": "1", "minWidth": "360px",
}

def section_title(text):
    return html.H3(text, style={
        "color": "#1a3c6b", "borderLeft": "4px solid #1a3c6b",
        "paddingLeft": "10px", "marginBottom": "8px", "fontSize": "15px",
    })

# ── Tab 1 layout: Notebook Figures ────────────────────────────
tab_figures = html.Div([
    html.Div([
        html.P(
            "All 5 interactive Plotly figures from Phase 2 (Task 1) — "
            "converted from Phase 1 static charts.",
            style={"color": "#555", "fontSize": "13px", "margin": "0 0 20px"},
        ),

        # Fig 1
        html.Div([
            section_title("Fig 1 — Rating Distribution (Histogram + Violin)"),
            dcc.Graph(figure=fig1, config={"displayModeBar": True}),
        ], style={"marginBottom": "24px"}),

        # Fig 2
        html.Div([
            section_title("Fig 2 — Top 10 Book Formats (Horizontal Bar)"),
            dcc.Graph(figure=fig2, config={"displayModeBar": True}),
        ], style={"marginBottom": "24px"}),

        # Fig 3
        html.Div([
            section_title("Fig 3 — Pages vs. Rating (Bubble Scatter)"),
            dcc.Graph(figure=fig3, config={"displayModeBar": True}),
        ], style={"marginBottom": "24px"}),

        # Fig 4
        html.Div([
            section_title("Fig 4 — Rating by Format (Multi-Trace Overlay Histogram)"),
            dcc.Graph(figure=fig4, config={"displayModeBar": True}),
        ], style={"marginBottom": "24px"}),

        # Fig 5
        html.Div([
            section_title("Fig 5 — Top Genres Animated by Format (Animated Bar)"),
            dcc.Graph(figure=fig5, config={"displayModeBar": True}),
        ], style={"marginBottom": "24px"}),

    ], style={"padding": "24px 30px", "background": "#f0f4fb", "minHeight": "100vh"}),
])

# ── Tab 2 layout: Interactive Dashboard ───────────────────────
tab_dashboard = html.Div([

    # Controls row
    html.Div([
        html.Div([
            html.Label("Book Format", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id="format-dropdown",
                options=[{"label": f, "value": f} for f in top_formats],
                value=top_formats[:3],
                multi=True,
                style={"fontSize": "13px"},
            ),
        ], style={"flex": "1", "minWidth": "280px"}),

        html.Div([
            html.Label("Rating Range", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.RangeSlider(
                id="rating-slider",
                min=0, max=5, step=0.1,
                value=[3.0, 5.0],
                marks={i: str(i) for i in range(6)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ], style={"flex": "2", "minWidth": "300px"}),

    ], style={
        "display": "flex", "gap": "24px", "flexWrap": "wrap",
        "padding": "20px 30px", "background": "white",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.07)", "alignItems": "flex-end",
    }),

    # KPI row
    html.Div(id="kpi-row", style={
        "display": "flex", "gap": "16px",
        "padding": "16px 30px", "background": "#fafbff", "flexWrap": "wrap",
    }),

    # Charts row
    html.Div([
        html.Div([
            html.H3("Top Genres by Book Count",
                    style={"fontSize": "15px", "marginBottom": "4px"}),
            dcc.Graph(id="genre-bar", config={"displayModeBar": True}),
        ], style=CHART_CARD_STYLE),

        html.Div([
            html.H3("Pages vs. Rating",
                    style={"fontSize": "15px", "marginBottom": "4px"}),
            dcc.Graph(id="scatter-chart", config={"displayModeBar": True}),
        ], style=CHART_CARD_STYLE),

    ], style={"display": "flex", "gap": "16px", "padding": "16px 30px", "flexWrap": "wrap"}),

], style={"background": "#f0f4fb", "minHeight": "100vh"})

# ── Full layout ────────────────────────────────────────────────
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("📚 GoodReads Dashboard",
                style={"margin": "0", "color": "white", "fontSize": "22px"}),
        html.P("Phase 2 — Interactive Analytics | GoodReads 100k Books",
               style={"margin": "4px 0 0", "color": "#cfe2ff", "fontSize": "13px"}),
    ], style={"background": "#1a3c6b", "padding": "20px 30px"}),

    # Tabs
    dcc.Tabs(id="main-tabs", value="tab-figures", children=[
        dcc.Tab(label="📊 Notebook Figures (Task 1)",
                value="tab-figures",
                style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
        dcc.Tab(label="🔍 Interactive Dashboard (Task 2)",
                value="tab-dashboard",
                style=TAB_STYLE, selected_style=TAB_SEL_STYLE),
    ]),

    html.Div(id="tab-content"),

    # Footer
    html.Div(
        "Student: Nour Eddin Abuazzam | ID: 202210576 | GoodReads 100k Dataset",
        style={
            "textAlign": "center", "padding": "14px", "color": "#666",
            "fontSize": "12px", "borderTop": "1px solid #ddd",
            "background": "white",
        },
    ),

], style={"fontFamily": "Arial", "background": "#f0f4fb", "minHeight": "100vh"})

# ─────────────────────────────────────────────────────────────
# 5) CALLBACKS
# ─────────────────────────────────────────────────────────────

# Tab switcher
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
)
def render_tab(tab):
    if tab == "tab-figures":
        return tab_figures
    return tab_dashboard


# Dashboard charts + KPIs
@app.callback(
    Output("genre-bar",    "figure"),
    Output("scatter-chart","figure"),
    Output("kpi-row",      "children"),
    Input("format-dropdown","value"),
    Input("rating-slider",  "value"),
)
def update(selected_formats, rating_range):

    if not selected_formats:
        selected_formats = top_formats[:3]

    low, high = rating_range

    filtered = df[
        df["bookformat"].isin(selected_formats) &
        (df["rating"] >= low) &
        (df["rating"] <= high)
    ]

    # ── KPI cards ────────────────────────────────────────
    def kpi_card(label, value, color="#1a3c6b"):
        return html.Div([
            html.P(label,  style={"margin": "0", "fontSize": "12px", "color": "#666"}),
            html.H2(value, style={"margin": "4px 0 0", "color": color, "fontSize": "22px"}),
        ], style=CARD_STYLE)

    avg_rating   = round(filtered["rating"].mean(), 2) if len(filtered) else 0
    median_pages = int(filtered["pages"].median())     if len(filtered) else 0

    kpis = [
        kpi_card("Books in Selection",  f"{len(filtered):,}",    "#1a3c6b"),
        kpi_card("Avg Rating",          f"{avg_rating:.2f} ⭐",  "#2e7d32"),
        kpi_card("Median Pages",        f"{median_pages:,}",     "#6a1b9a"),
        kpi_card("Formats Selected",    str(len(selected_formats)), "#c62828"),
    ]

    # ── Genre bar ────────────────────────────────────────
    gc = filtered["genre"].value_counts().head(10).reset_index()
    gc.columns = ["Genre", "Books"]
    gc["avg_rating"] = gc["Genre"].map(
        filtered.groupby("genre")["rating"].mean().round(3)
    )

    bar_fig = px.bar(
        gc, x="Books", y="Genre", orientation="h",
        color="Books", color_continuous_scale="Blues",
        text="Books",
        custom_data=["Genre", "avg_rating"],
    )
    bar_fig.update_traces(
        texttemplate="%{x:,}", textposition="outside",
        hovertemplate="<b>%{y}</b><br>Books: %{x:,}<br>Avg Rating: %{customdata[1]:.2f}<extra></extra>",
    )
    bar_fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        plot_bgcolor="#f9f9f9",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )

    # ── Scatter ──────────────────────────────────────────
    if len(filtered) > 0:
        scat = filtered.sample(min(3000, len(filtered)), random_state=42).copy()
        scat["log_tr"] = np.log1p(scat["totalratings"])

        scatter = px.scatter(
            scat, x="pages", y="rating",
            color="bookformat", size="log_tr", size_max=16,
            opacity=0.55,
            hover_data={"title": True, "author": True,
                        "pages": True, "rating": ":.2f",
                        "totalratings": True, "log_tr": False},
            labels={"pages": "Pages", "rating": "Rating",
                    "bookformat": "Format", "totalratings": "Total Ratings"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        scatter.update_layout(
            plot_bgcolor="#f9f9f9", legend_title="Format",
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
        )
    else:
        scatter = px.scatter(title="No data for selected filters")

    return bar_fig, scatter, kpis


# ─────────────────────────────────────────────────────────────
# 6) RUN (LOCAL + RENDER SAFE)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
