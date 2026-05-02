import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# ── Data loading & cleaning ───────────────────────────────────────────────────
df = pd.read_csv("GoodReads_100k_books.csv")
df["title"]        = df["title"].fillna("Unknown")
df = df.dropna(subset=["bookformat", "genre"]).copy()
df["genre"]        = df["genre"].str.split(",").str[0].str.strip()
df["bookformat"]   = df["bookformat"].str.strip().str.title()
df["totalratings"] = pd.to_numeric(df["totalratings"], errors="coerce")
df["rating"]       = pd.to_numeric(df["rating"],       errors="coerce")
df["pages"]        = pd.to_numeric(df["pages"],        errors="coerce")
df = df.dropna(subset=["rating", "pages", "totalratings"]).copy()
df = df[df["pages"] < 3000].copy()

top_formats = df["bookformat"].value_counts().head(6).index.tolist()

# ── App layout ────────────────────────────────────────────────────────────────
app = Dash(__name__)
server = app.server   # required for Render / gunicorn

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("📚 GoodReads Interactive Dashboard",
                style={"margin": "0", "color": "white", "fontFamily": "Arial"}),
        html.P("Phase 2 — Data Exploration and Visualization (606475)",
               style={"margin": "4px 0 0", "color": "#cfe2ff", "fontSize": "14px"})
    ], style={"background": "#1a3c6b", "padding": "20px 30px"}),

    # ── Controls row ─────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Book Format", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Dropdown(
                id="format-dropdown",
                options=[{"label": f, "value": f} for f in top_formats],
                value=top_formats[:3],
                multi=True,
                placeholder="Select format(s)…",
                style={"fontSize": "13px"}
            )
        ], style={"flex": "1", "minWidth": "260px", "marginRight": "30px"}),

        html.Div([
            html.Label("Rating Range", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.RangeSlider(
                id="rating-slider",
                min=0, max=5, step=0.1,
                value=[3.0, 5.0],
                marks={i: str(i) for i in range(0, 6)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={"flex": "2", "minWidth": "280px"})

    ], style={"display": "flex", "alignItems": "flex-end",
              "padding": "20px 30px", "background": "#f4f8ff",
              "borderBottom": "1px solid #d0d8e8", "flexWrap": "wrap", "gap": "10px"}),

    # ── KPI cards (callback-driven) ───────────────────────────────────────────
    html.Div(id="kpi-row",
             style={"display": "flex", "gap": "16px", "padding": "16px 30px",
                    "background": "#fafbff", "flexWrap": "wrap"}),

    # ── Charts row ────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H3("Top Genres by Book Count",
                    style={"fontSize": "15px", "marginBottom": "4px"}),
            dcc.Graph(id="genre-bar", config={"displayModeBar": True})
        ], style={"flex": "1", "minWidth": "360px", "background": "white",
                  "borderRadius": "8px", "padding": "16px",
                  "boxShadow": "0 1px 6px rgba(0,0,0,0.1)"}),

        html.Div([
            html.H3("Pages vs. Rating — Bubble = log(Total Ratings)",
                    style={"fontSize": "15px", "marginBottom": "4px"}),
            dcc.Graph(id="scatter-chart", config={"displayModeBar": True})
        ], style={"flex": "1", "minWidth": "360px", "background": "white",
                  "borderRadius": "8px", "padding": "16px",
                  "boxShadow": "0 1px 6px rgba(0,0,0,0.1)"})

    ], style={"display": "flex", "gap": "16px", "padding": "16px 30px",
              "flexWrap": "wrap"}),

    # Footer
    html.Div(
        "Student: Nour Eddin Abuazzam | ID: 202210576 | GoodReads 100k Dataset",
        style={"textAlign": "center", "padding": "14px", "color": "#666",
               "fontSize": "12px", "borderTop": "1px solid #ddd"}
    )

], style={"fontFamily": "Arial", "background": "#f0f4fb", "minHeight": "100vh"})


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output("genre-bar",     "figure"),
    Output("scatter-chart", "figure"),
    Output("kpi-row",       "children"),
    Input("format-dropdown", "value"),
    Input("rating-slider",   "value"),
)
def update_charts(selected_formats, rating_range):
    if not selected_formats:
        selected_formats = top_formats[:3]
    low, high = rating_range

    filtered = df[
        (df["bookformat"].isin(selected_formats)) &
        (df["rating"] >= low) &
        (df["rating"] <= high)
    ]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    def kpi_card(label, value, color):
        return html.Div([
            html.P(label, style={"margin": "0", "fontSize": "12px", "color": "#666"}),
            html.H2(value, style={"margin": "4px 0 0", "color": color, "fontSize": "22px"})
        ], style={"background": "white", "borderRadius": "8px",
                  "padding": "12px 20px",
                  "boxShadow": "0 1px 4px rgba(0,0,0,0.08)", "minWidth": "130px"})

    kpis = [
        kpi_card("Books in Selection", f"{len(filtered):,}",              "#1a3c6b"),
        kpi_card("Avg Rating",         f"{filtered['rating'].mean():.3f}", "#2e7d32"),
        kpi_card("Median Pages",       f"{int(filtered['pages'].median()):,}", "#6a1b9a"),
        kpi_card("Formats Selected",   str(len(selected_formats)),         "#c62828"),
    ]

    # ── Genre bar ──────────────────────────────────────────────────────────────
    gc = (filtered["genre"].value_counts()
                           .head(10)
                           .reset_index())
    gc.columns = ["Genre", "Books"]
    gc["avg_rating"] = gc["Genre"].map(
        filtered.groupby("genre")["rating"].mean().round(3))

    bar_fig = px.bar(
        gc, x="Books", y="Genre", orientation="h",
        color="Books", color_continuous_scale="Blues",
        text="Books",
        hover_data={"Genre": True, "Books": True, "avg_rating": ":.3f"},
        labels={"avg_rating": "Avg Rating"}
    )
    bar_fig.update_traces(
        texttemplate="%{x:,}", textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>Books: %{x:,}<br>"
            "Avg Rating: %{customdata[0]:.2f}<extra></extra>"
        )
    )
    bar_fig.update_layout(
        yaxis=dict(autorange="reversed"),
        coloraxis_showscale=False,
        plot_bgcolor="#f9f9f9",
        margin=dict(l=10, r=30, t=10, b=10),
        height=380
    )

    # ── Scatter ────────────────────────────────────────────────────────────────
    scat = filtered.sample(min(3000, len(filtered)), random_state=42).copy()
    scat["log_tr"] = np.log1p(scat["totalratings"])

    scat_fig = px.scatter(
        scat,
        x="pages", y="rating",
        color="bookformat",
        size="log_tr", size_max=16,
        opacity=0.55,
        hover_data={"title": True, "author": True, "pages": True,
                    "rating": ":.2f", "totalratings": ":,", "log_tr": False},
        labels={"pages": "Pages", "rating": "Rating",
                "bookformat": "Format", "totalratings": "Total Ratings"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    scat_fig.update_layout(
        plot_bgcolor="#f9f9f9",
        legend_title="Format",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380
    )

    return bar_fig, scat_fig, kpis


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
