"""
Feature engineering for Lead Score Flow.
Generates rich behavioral, intent, engagement, frequency, recency,
and metadata features for ML model training and scoring.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List
import yaml


class FeatureEngineer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Scalers + encoders stored for reuse during inference
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []

    # ---------------------------------------------------------
    # Feature engineering core logic
    # ---------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()

        # --------------------------
        # Behavioral engagement metrics
        # --------------------------
        df["engagement_score"] = (
            df["email_opens"] * 0.1
            + df["email_clicks"] * 0.2
            + df["website_visits"] * 0.15
            + (df["time_on_site"] / 60) * 0.05
            + df["pages_viewed"] * 0.1
            + df["content_downloads"] * 0.3
        )

        # --------------------------
        # Intent score (strong signals)
        # --------------------------
        df["intent_score"] = (
            df["demo_requests"] * 3.0
            + df["pricing_page_views"] * 2.0
            + df["form_submissions"] * 1.5
            + df["event_attendance"] * 1.0
        )

        # --------------------------
        # Recency features
        # --------------------------
        df["recency_score"] = 1 / (df["last_interaction_days"] + 1)
        df["is_recent"] = (df["last_interaction_days"] <= 7).astype(int)

        # --------------------------
        # Frequency features
        # --------------------------
        df["avg_interactions_per_day"] = df["total_interactions"] / (
            df["days_since_first_interaction"] + 1
        )
        df["high_frequency"] = (
            df["interaction_frequency"] > df["interaction_frequency"].median()
        ).astype(int)

        # --------------------------
        # Engagement depth
        # --------------------------
        df["engagement_depth"] = df["pages_viewed"] * df["time_on_site"]
        df["engagement_consistency"] = df["email_opens"] / (
            df["website_visits"] + 1
        )

        # --------------------------
        # Firmographic encodings
        # --------------------------
        seniority_map = {"Entry": 1, "Mid": 2, "Senior": 3, "Executive": 4}
        df["seniority_numeric"] = df["seniority_level"].map(seniority_map)

        size_map = {"1-10": 1, "11-50": 2, "51-200": 3, "201-1000": 4, "1000+": 5}
        df["company_size_numeric"] = df["company_size"].map(size_map)

        # --------------------------
        # Decision maker flag
        # --------------------------
        df["is_decision_maker"] = df["job_title"].isin(
            ["Director", "VP", "C-Level"]
        ).astype(int)

        # --------------------------
        # Industry encoding
        # --------------------------
        if fit:
            self.encoders["industry"] = LabelEncoder()
            df["industry_encoded"] = self.encoders["industry"].fit_transform(
                df["industry"]
            )
        else:
            df["industry_encoded"] = df["industry"].apply(
                lambda x: self.encoders["industry"].transform([x])[0]
                if x in self.encoders["industry"].classes_
                else -1
            )

        # --------------------------
        # Market segmentation
        # --------------------------
        major_markets = ["USA", "UK", "Canada", "Germany"]
        df["is_major_market"] = df["country"].isin(major_markets).astype(int)

        # --------------------------
        # Additional behavioral signals
        # --------------------------
        df["content_engagement_ratio"] = df["content_downloads"] / (
            df["website_visits"] + 1
        )
        df["has_demo_request"] = (df["demo_requests"] > 0).astype(int)
        df["pricing_interest"] = (df["pricing_page_views"] > 0).astype(int)

        return df

    # ---------------------------------------------------------
    # Master feature list
    # ---------------------------------------------------------
    def get_feature_list(self):

        raw_features = [
            'email_opens', 'email_clicks', 'website_visits',
            'time_on_site', 'pages_viewed', 'content_downloads',
            'form_submissions', 'event_attendance', 'demo_requests',
            'pricing_page_views', 'feature_page_views',
            'days_since_first_interaction', 'total_interactions',
            'last_interaction_days', 'interaction_frequency'
        ]

        engineered_features = [
            'engagement_score', 'intent_score', 'recency_score',
            'avg_interactions_per_day', 'engagement_depth', 
            'engagement_consistency', 'seniority_numeric',
            'company_size_numeric', 'industry_encoded',
            'content_engagement_ratio'
        ]

        binary_features = [
            'is_recent', 'high_frequency', 'is_decision_maker',
            'is_major_market', 'has_demo_request', 'pricing_interest'
        ]

        # Features to scale (exclude binary flags)
        numeric_to_scale = raw_features + engineered_features

        return numeric_to_scale + binary_features

    # ---------------------------------------------------------
    # Scaling (fit during training, reuse during inference)
    # ---------------------------------------------------------
    def scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        feature_cols = self.get_feature_list()

        # Define binary features again to exclude them from scaling
        binary_features = [
            'is_recent', 'high_frequency', 'is_decision_maker',
            'is_major_market', 'has_demo_request', 'pricing_interest'
        ]

        numeric_cols = [c for c in feature_cols if c not in binary_features]

        if fit:
            self.scalers['standard'] = StandardScaler()
            df[numeric_cols] = self.scalers['standard'].fit_transform(df[numeric_cols])
        else:
            df[numeric_cols] = self.scalers['standard'].transform(df[numeric_cols])

        # Ensure binary columns are int
        for col in binary_features:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    # ---------------------------------------------------------
    # Main pipeline for training + inference
    # ---------------------------------------------------------
    def prepare_data(self, df: pd.DataFrame, fit: bool = False, 
                     scale: bool = True) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete feature preparation pipeline
        
        Args:
            df: Input dataframe
            fit: Whether to fit transformers
            scale: Whether to scale features
        
        Returns:
            Processed dataframe and feature names
        """
        # Engineer features
        df = self.engineer_features(df, fit=fit)
        
        # Get feature list
        feature_cols = self.get_feature_list()
        
        # Scale if requested
        if scale:
            df = self.scale_features(df, fit=fit)
        
        self.feature_names = feature_cols
        
        return df, feature_cols
