"""UI components and Apple-inspired design system for Streamlit AI Lab."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def inject_apple_theme():
    """Inject custom CSS for Apple-inspired aesthetic and UX."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        /* Main typography & font stack */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif !important;
            letter-spacing: -0.01em;
        }

        /* Hero Header Banner */
        .hero-banner {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
        }

        .hero-title {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            line-height: 1.2;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #94A3B8;
            margin-top: 6px;
            font-weight: 400;
        }

        /* Status Badge Pills */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
            border-radius: 100px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-right: 8px;
        }

        .status-success {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        .status-info {
            background: rgba(59, 130, 246, 0.15);
            color: #60A5FA;
            border: 1px solid rgba(96, 165, 250, 0.3);
        }

        .status-warning {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            border: 1px solid rgba(251, 191, 36, 0.3);
        }

        /* Metric Cards (iOS Control Center Widget Style) */
        .metric-card {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 16px;
            padding: 20px 24px;
            text-align: left;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(59, 130, 246, 0.4);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.2);
        }

        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.1;
        }

        .metric-lbl {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        .metric-sub {
            font-size: 0.8rem;
            color: #64748B;
            margin-top: 6px;
        }

        /* Apple Segmented Control Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.6);
            padding: 6px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 600;
            font-size: 0.95rem;
            color: #94A3B8;
            border: none !important;
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2563EB !important;
            color: #FFFFFF !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }

        /* Result Cards */
        .result-card {
            background: linear-gradient(135deg, rgba(30, 58, 138, 0.3) 0%, rgba(15, 23, 42, 0.6) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 18px;
            padding: 24px;
            margin-top: 16px;
            margin-bottom: 24px;
        }

        /* Streamlit Input & Form Styling Overrides */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 10px 24px !important;
            transition: all 0.2s ease !important;
        }

        .stDownloadButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
        }

        /* Source snippet card */
        .source-card {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #3B82F6;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_global_importance_chart(df: pd.DataFrame, method_name: str) -> alt.Chart:
    """Create a sleek horizontal bar chart for global feature importance."""
    chart_df = df.copy()
    chart_df["importance_pct"] = chart_df["importance"] * 100

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=24)
        .encode(
            x=alt.X("importance_pct:Q", title="Importancia (%)", axis=alt.Axis(grid=True, gridColor="rgba(255,255,255,0.05)")),
            y=alt.Y("feature:N", title=None, sort="-x", axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            color=alt.Color(
                "importance_pct:Q",
                scale=alt.Scale(scheme="tealblues"),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Característica"),
                alt.Tooltip("importance_pct:Q", title="Importancia (%)", format=".1f"),
            ],
        )
        .properties(height=240, title=alt.Title(f"Importancia Global de Características ({method_name})", fontSize=14, color="#94A3B8"))
        .configure_view(strokeWidth=0)
    )
    return chart


def create_local_explanation_chart(df: pd.DataFrame, method_name: str) -> alt.Chart:
    """Create a bi-directional horizontal bar chart for local feature impact (positive vs negative)."""
    chart_df = df.copy()
    chart_df["impact_pct"] = chart_df["impact"] * 100
    chart_df["efecto"] = chart_df["impact"].apply(lambda x: "Favorece Renovación" if x >= 0 else "Desfavorece Renovación")

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadius=4, size=22)
        .encode(
            x=alt.X("impact_pct:Q", title="Impacto en la Probabilidad (%)", axis=alt.Axis(grid=True, gridColor="rgba(255,255,255,0.05)")),
            y=alt.Y("feature:N", title=None, sort="-x", axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            color=alt.Color(
                "efecto:N",
                scale=alt.Scale(domain=["Favorece Renovación", "Desfavorece Renovación"], range=["#10B981", "#EF4444"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Característica"),
                alt.Tooltip("impact_pct:Q", title="Impacto (%)", format="+.2f"),
                alt.Tooltip("efecto:N", title="Efecto"),
            ],
        )
        .properties(height=240, title=alt.Title(f"Explicación Local de Predicción ({method_name})", fontSize=14, color="#94A3B8"))
        .configure_view(strokeWidth=0)
    )
    return chart


def create_vision_probabilities_chart(probabilities: dict, winner_label: str) -> alt.Chart:
    """Create a styled horizontal bar chart for top class probabilities in vision classifier."""
    data = []
    for cls, prob in probabilities.items():
        data.append({
            "Clase": cls,
            "Probabilidad (%)": prob * 100,
            "EsGanador": "Clase Predicha" if cls == winner_label else "Otra Clase",
        })
    df = pd.DataFrame(data).sort_values("Probabilidad (%)", ascending=False)

    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, size=24)
        .encode(
            x=alt.X("Probabilidad (%):Q", title="Confianza Relativa (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(grid=True, gridColor="rgba(255,255,255,0.05)")),
            y=alt.Y("Clase:N", title=None, sort="-x", axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            color=alt.Color(
                "EsGanador:N",
                scale=alt.Scale(domain=["Clase Predicha", "Otra Clase"], range=["#3B82F6", "#475569"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("Clase:N"),
                alt.Tooltip("Probabilidad (%):Q", format=".1f"),
            ],
        )
        .properties(height=220, title=alt.Title("Top-5 Clases Predichas (Distribución de Confianza)", fontSize=14, color="#94A3B8"))
        .configure_view(strokeWidth=0)
    )
    return chart


def create_donut_gauge(probability: float, title: str = "Probabilidad de Renovación") -> alt.Chart:
    """Create an Apple-style donut gauge chart for probability percentage."""
    pct = max(0.0, min(1.0, probability))
    data = pd.DataFrame([
        {"category": "Valor", "value": pct * 100, "color": "#10B981" if pct >= 0.5 else "#EF4444"},
        {"category": "Restante", "value": (1.0 - pct) * 100, "color": "rgba(255,255,255,0.08)"},
    ])

    chart = (
        alt.Chart(data)
        .mark_arc(innerRadius=60, outerRadius=90, cornerRadius=6)
        .encode(
            theta=alt.Theta(field="value", type="quantitative"),
            color=alt.Color(field="color", type="nominal", scale=None),
            tooltip=[alt.Tooltip("category:N", title="Estado"), alt.Tooltip("value:Q", title="Porcentaje", format=".1f")],
        )
        .properties(width=200, height=200, title=alt.Title(title, fontSize=13, color="#94A3B8"))
    )
    return chart
