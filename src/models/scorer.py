"""
Hybrid lead scoring system combining ML model probability +
rule-based scoring.
"""

import numpy as np
import pandas as pd
import joblib
import yaml


class LeadScorer:
    def __init__(self, model_path="models/lead_scorer.pkl", config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        data = joblib.load(model_path)
        self.model = data["model"]
        self.model_version = data["model_version"]

        self.score_ranges = self.config["model"]["score_ranges"]

    # ---------------------------------------------------------
    # ML score
    # ---------------------------------------------------------
    def predict_probability(self, X):
        return self.model.predict_proba(X)[:, 1]

    # ---------------------------------------------------------
    # Rule-based score
    # ---------------------------------------------------------
    def rule_score(self, df: pd.DataFrame):

        score = (
            df["demo_requests"] * 25
            + df["pricing_page_views"] * 15
            + (df["is_decision_maker"] & df["is_recent"]) * 10
            + (df["form_submissions"] > 0).astype(int) * 8
            + (
                (df["seniority_numeric"] >= 3)
                & (df["company_size_numeric"] >= 4)
            ).astype(int)
            * 12
            + df["high_frequency"] * 5
            + (df["total_interactions"] > 20).astype(int) * 7
            + (df["is_recent"] & (df["website_visits"] > 5)).astype(int) * 6
        )

        if score.max() > 0:
            score = (score / score.max()) * 100

        return score

    # ---------------------------------------------------------
    # Hybrid scoring
    # ---------------------------------------------------------
    def hybrid_score(self, df, X, ml_weight=0.7):
        ml = self.predict_probability(X) * 100
        rule = self.rule_score(df)

        final = (ml_weight * ml) + ((1 - ml_weight) * rule)
        return np.clip(final, 0, 100)

    # ---------------------------------------------------------
    # Category mapping
    # ---------------------------------------------------------
    def categorize(self, score):
        for category, (min_s, max_s) in self.score_ranges.items():
            if min_s <= score <= max_s:
                return category
        return "unqualified"

    # ---------------------------------------------------------
    # Full scoring pipeline
    # ---------------------------------------------------------
    def score_leads(self, df, X, method="hybrid"):
        if method == "ml":
            score = self.predict_probability(X) * 100
        elif method == "rule":
            score = self.rule_score(df)
        else:
            score = self.hybrid_score(df, X)

        df = df.copy()
        df["lead_score"] = score
        df["score_category"] = df["lead_score"].apply(self.categorize)
        df["score_rank"] = df["lead_score"].rank(ascending=False, method="min").astype(int)
        df["model_version"] = self.model_version

        return df

    # ---------------------------------------------------------
    # Insights for BI agent
    # ---------------------------------------------------------
    def insights(self, df: pd.DataFrame):
        return {
            "total_leads": len(df),
            "avg_score": float(df["lead_score"].mean()),
            "median_score": float(df["lead_score"].median()),
            "std_score": float(df["lead_score"].std()),
            "hot_leads": int((df["score_category"] == "hot").sum()),
            "warm_leads": int((df["score_category"] == "warm").sum()),
            "cold_leads": int((df["score_category"] == "cold").sum()),
            "category_distribution": df["score_category"].value_counts().to_dict(),
        }
