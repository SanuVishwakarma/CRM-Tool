"""
Lead Score Flow — Full Agentic Pipeline (Groq LLM + ML + PostgreSQL)
"""

import os
import pandas as pd
from datetime import datetime

from src.models.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.scorer import LeadScorer
from src.crew.lead_scoring_crew import LeadScoringCrew
from src.database.db_manager import DatabaseManager


def print_header(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72 + "\n")


def main():
    start_time = datetime.now()

    print_header("🎯 LEAD SCORE FLOW — AGENTIC AI PIPELINE (GROQ + ML + POSTGRESQL)")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ============================================================
    # PHASE 1: LOAD DATA
    # ============================================================
    print_header("PHASE 1 — Load Raw Training Data")

    train_path = "data/raw/historical_leads.csv"
    new_path = "data/raw/new_leads.csv"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    if not os.path.exists(new_path):
        raise FileNotFoundError(f"New leads data not found: {new_path}")

    train_df = pd.read_csv(train_path)
    new_df = pd.read_csv(new_path)

    print(f"✓ Loaded {len(train_df)} training samples")
    print(f"✓ Loaded {len(new_df)} new leads")
    print(f"Training conversion rate: {train_df['converted'].mean():.2%}")

    # ============================================================
    # PHASE 2: FEATURE ENGINEERING
    # ============================================================
    print_header("PHASE 2 — Feature Engineering")

    print_header("Preparing Required Columns")

    def ensure_required_columns(df):
        """Adds required columns before FeatureEngineering pipeline"""

        # interaction_frequency (mandatory)
        df["interaction_frequency"] = (
            df["total_interactions"] / (df["days_since_first_interaction"] + 1)
        )

        # Optional-but-safe fields (if missing)
        defaults = {
            "time_on_site": 0,
            "pages_viewed": 0,
            "content_downloads": 0,
            "form_submissions": 0,
            "event_attendance": 0,
            "demo_requests": 0,
            "pricing_page_views": 0,
            "feature_page_views": 0,
        }

        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        return df
    
    train_df = ensure_required_columns(train_df)
    new_df = ensure_required_columns(new_df)

    fe = FeatureEngineer()

    print("🔧 Engineering features (training)...")
    train_df, feature_cols = fe.prepare_data(train_df, fit=True, scale=True)
    X_train = train_df[feature_cols]
    y_train = train_df["converted"]

    print("🔧 Engineering features (new leads)...")
    new_df, _ = fe.prepare_data(new_df, fit=False, scale=True)
    X_new = new_df[feature_cols]

    print("✓ Feature engineering completed")
    print(f"Total features created: {len(feature_cols)}")

    # ============================================================
    # PHASE 3: DATA ANALYSIS AGENT (Groq)
    # ============================================================
    print_header("PHASE 3 — CrewAI Data Analyst (Groq)")

    crew = LeadScoringCrew()

    summary = crew.run_analysis_workflow(train_df, has_labels=True)

    with open("reports/data_analysis_report.txt", "w") as f:
        f.write(summary)

    print("✓ Data analysis report saved")

    # ============================================================
    # PHASE 4: TRAIN ML MODEL
    # ============================================================
    print_header("PHASE 4 — ML Model Training")

    trainer = ModelTrainer()

    print("🚀 Training model...")
    metrics = trainer.train(X_train, y_train, use_smote=True)

    print("\nModel Metrics:")
    print(metrics)

    print("\n💾 Saving model to models/lead_scorer.pkl ...")
    trainer.save_model()

    # ============================================================
    # PHASE 5: ML ENGINEER AGENT (Groq)
    # ============================================================
    print_header("PHASE 5 — CrewAI ML Engineer (Groq)")

    modeling_report = crew.run_modeling_workflow(
        metrics, train_size=len(X_train), test_size=int(len(X_train) * 0.2)
    )

    with open("reports/modeling_report.txt", "w") as f:
        f.write(modeling_report)

    print("✓ Model evaluation report saved")

    # ============================================================
    # PHASE 6: SCORE NEW LEADS
    # ============================================================
    print_header("PHASE 6 — Scoring New Leads")

    scorer = LeadScorer()

    scored_df = scorer.score_leads(new_df, X_new, method="hybrid")

    print(f"✓ Scored {len(scored_df)} leads")
    print(scored_df["score_category"].value_counts())

    insights = scorer.insights(scored_df)

    print("\n📊 Lead Insights:")
    print(insights)

    os.makedirs("data/scored", exist_ok=True)
    scored_df.to_csv("data/scored/scored_leads.csv", index=False)
    print("✓ Scored leads saved")

    # ============================================================
    # PHASE 7: SCORING ANALYSIS (Groq)
    # ============================================================
    print_header("PHASE 7 — CrewAI Scoring Specialist")

    scoring_report = crew.run_scoring_workflow(
        total_leads=len(scored_df),
        model_version=scorer.model_version,
        method="hybrid",
    )

    with open("reports/scoring_report.txt", "w") as f:
        f.write(scoring_report)

    print("✓ Scoring analysis saved")

    # ============================================================
    # PHASE 8: DATABASE STORAGE
    # ============================================================
    print_header("PHASE 8 — Save Results to PostgreSQL")

    db = DatabaseManager()

    print("💾 Storing model metadata...")
    db.insert_model_metadata({
        "model_version": metrics["model_version"],
        "algorithm": metrics["algorithm"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "roc_auc": metrics["roc_auc"],
        "features_used": metrics["features_used"],
    })

    print("💾 Storing leads + scores...")
    for _, row in scored_df.iterrows():
        lead_data = {
            "lead_id": row["lead_id"],
            "email": row.get("email"),
            "company_name": row.get("company_name"),
            "industry": row["industry"],
            "job_title": row["job_title"],
            "seniority_level": row["seniority_level"],
            "company_size": row["company_size"],
            "country": row["country"],
            "created_at": datetime.now(),
        }

        db.insert_lead(lead_data)
        db.insert_score(
            row["lead_id"],
            float(row["lead_score"]),
            row["score_category"],
            scorer.model_version,
            str(list(feature_cols)),
        )

    print(f"✓ Stored {len(scored_df)} leads in PostgreSQL")

    # ============================================================
    # PHASE 9: BI REPORTING (Groq)
    # ============================================================
    print_header("PHASE 9 — CrewAI BI Analyst (Groq)")

    bi_report = crew.run_reporting_workflow(insights, metrics)

    with open("reports/executive_report.txt", "w") as f:
        f.write(bi_report)

    print("✓ Executive BI report saved")

    # ============================================================
    # SUMMARY
    # ============================================================
    end_time = datetime.now()

    print_header("✅ PIPELINE COMPLETE")
    print("SUMMARY:")
    print(f"  • Training samples: {len(train_df)}")
    print(f"  • New leads scored: {len(scored_df)}")
    print(f"  • Hot leads: {insights['hot_leads']}")
    print(f"  • Model version: {scorer.model_version}")
    print(f"  • Duration: {end_time - start_time}")

    print("\n🎯 Next Steps:")
    print("  1. Run: uvicorn src.api.main:app --reload")
    print("  2. Run dashboard: streamlit run src/dashboard/app.py")
    print("  3. Review: reports/*.txt")
    print("  4. View database: AWS RDS console")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
