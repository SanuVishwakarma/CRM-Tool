"""
Streamlit Dashboard for Lead Score Flow (Groq + PostgreSQL Version)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from models.scorer import LeadScorer

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Lead Score Flow Dashboard",
    page_icon="🎯",
    layout="wide",
)

st.markdown(
    """
    <style>
        .metric-card {
            padding: 20px;
            border-radius: 8px;
            background-color: #f7f7f9;
            border: 1px solid #dedede;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Instantiate DB + Scorer
# ---------------------------------------------------------
db = DatabaseManager()
scorer = LeadScorer()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🎯 Lead Score Flow")
st.sidebar.subheader("Navigation")
page = st.sidebar.radio(
    "",
    ["Dashboard Overview", "Lead Explorer", "Model Performance", "Top Leads"],
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Model Version:**  {scorer.model_version}")


# ---------------------------------------------------------
# 1. Dashboard Overview
# ---------------------------------------------------------
if page == "Dashboard Overview":
    st.title("📊 Lead Score Flow — Insights Dashboard")

    recent_scores = db.get_recent_scores(limit=1000)
    distribution = db.get_score_distribution()

    if recent_scores.empty:
        st.warning("No score data available. Run scoring pipeline first.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Leads Scored", f"{len(recent_scores)}")

    with col2:
        st.metric("Average Score", f"{recent_scores['score'].mean():.2f}")

    with col3:
        hot = (recent_scores["score_category"] == "hot").sum()
        st.metric("🔥 Hot Leads", hot)

    with col4:
        warm = (recent_scores["score_category"] == "warm").sum()
        st.metric("♨️ Warm Leads", warm)

    st.markdown("---")

    # Score Distribution Chart
    st.subheader("📈 Score Distribution")
    fig = px.histogram(
        recent_scores,
        x="score",
        nbins=30,
        labels={"score": "Lead Score"},
        title="Distribution of Lead Scores",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Category Breakdown Pie Chart
    st.subheader("🎯 Category Breakdown")
    category_counts = recent_scores["score_category"].value_counts()

    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Score Categories",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Hot Leads by Industry
    st.subheader("🏢 Top Industries (Hot Leads)")

    hot_df = recent_scores[recent_scores["score_category"] == "hot"]

    if hot_df.empty:
        st.info("No hot leads available.")
    else:
        top_industries = hot_df["industry"].value_counts().head(10)
        fig = px.bar(
            x=top_industries.values,
            y=top_industries.index,
            orientation="h",
            labels={"x": "Number of Leads", "y": "Industry"},
            title="Top 10 Industries (Hot Leads)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 2. Lead Explorer
# ---------------------------------------------------------
elif page == "Lead Explorer":
    st.title("🔍 Lead Explorer")

    df = db.get_recent_scores(limit=1000)

    if df.empty:
        st.warning("No leads found.")
        st.stop()

    industries = ["All"] + sorted(df["industry"].unique().tolist())
    categories = ["All"] + sorted(df["score_category"].unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        selected_industry = st.selectbox("Filter by Industry", industries)

    with col2:
        selected_category = st.selectbox("Filter by Category", categories)

    filtered = df.copy()

    if selected_industry != "All":
        filtered = filtered[filtered["industry"] == selected_industry]

    if selected_category != "All":
        filtered = filtered[filtered["score_category"] == selected_category]

    st.subheader(f"📋 Leads ({len(filtered)})")

    st.dataframe(
        filtered[
            [
                "lead_id",
                "email",
                "company_name",
                "industry",
                "score",
                "score_category",
                "scored_at",
            ]
        ],
        use_container_width=True,
        height=450,
    )

    csv = filtered.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"leads_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# 3. Model Performance
# ---------------------------------------------------------
elif page == "Model Performance":
    st.title("🤖 Model Performance Report")

    perf = db.get_model_performance()

    if perf.empty:
        st.warning("No model performance logs found.")
        st.stop()

    latest = perf.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{latest['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{latest['precision_score']:.4f}")
    with col3:
        st.metric("Recall", f"{latest['recall']:.4f}")
    with col4:
        st.metric("F1 Score", f"{latest['f1_score']:.4f}")

    st.markdown("---")

    st.subheader("📈 Model Metrics Over Time")
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=perf["trained_at"],
            y=perf["accuracy"],
            mode="lines+markers",
            name="Accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=perf["trained_at"],
            y=perf["f1_score"],
            mode="lines+markers",
            name="F1 Score",
        )
    )

    fig.update_layout(
        xaxis_title="Training Time",
        yaxis_title="Score",
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# 4. Top Leads
# ---------------------------------------------------------
elif page == "Top Leads":
    st.title("🏆 Top Leads")

    df = db.get_recent_scores(limit=1000)

    if df.empty:
        st.warning("No leads found.")
        st.stop()

    num = st.slider("Number of Top Leads", 5, 100, 10)

    top_df = df.nlargest(num, "score")

    for idx, row in top_df.iterrows():
        with st.expander(
            f"{row['company_name']} — Score: {row['score']:.1f} ({row['score_category'].upper()})"
        ):
            st.write(f"**Lead ID:** {row['lead_id']}")
            st.write(f"**Email:** {row['email']}")
            st.write(f"**Industry:** {row['industry']}")
            st.write(f"**Score:** {row['score']:.2f}")
            st.write(f"**Category:** {row['score_category']}")
            st.write(f"**Scored At:** {row['scored_at']}")

    csv = top_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Top Leads",
        data=csv,
        file_name="top_leads.csv",
        mime="text/csv",
    )
