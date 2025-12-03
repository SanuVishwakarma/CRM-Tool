"""
PostgreSQL Database Manager for Lead Score Flow.
Compatible with AWS RDS and environment variable configuration.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()


class DatabaseManager:
    """
    Handles PostgreSQL connection, table creation, and CRUD operations
    for leads, scores, and model metadata.
    """

    def __init__(self):
        self.engine = self._init_engine()
        self._create_tables()

    # ---------------------------------------------------------
    # Build DSN dynamically from environment variables
    # ---------------------------------------------------------
    def _init_engine(self):
        try:
            host = os.getenv("POSTGRES_HOST")
            port = os.getenv("POSTGRES_PORT", "5432")
            user = os.getenv("POSTGRES_USER")
            password = os.getenv("POSTGRES_PASSWORD")
            database = os.getenv("POSTGRES_DB")
            sslmode = os.getenv("POSTGRES_SSLMODE", "require")

            if not all([host, port, user, password, database]):
                raise ValueError("PostgreSQL environment variables are missing!")

            dsn = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
                f"?sslmode={sslmode}"
            )

            return create_engine(dsn, pool_pre_ping=True)

        except Exception as e:
            raise RuntimeError(f"Database connection error: {e}")

    # ---------------------------------------------------------
    # Test connection
    # ---------------------------------------------------------
    def test_connection(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ---------------------------------------------------------
    # Create tables if missing
    # ---------------------------------------------------------
    def _create_tables(self):
        create_leads_table = """
        CREATE TABLE IF NOT EXISTS leads (
            lead_id VARCHAR(255) PRIMARY KEY,
            email VARCHAR(255),
            company_name VARCHAR(255),
            industry VARCHAR(255),
            job_title VARCHAR(255),
            seniority_level VARCHAR(255),
            company_size VARCHAR(50),
            country VARCHAR(255),
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        );
        """

        create_scores_table = """
        CREATE TABLE IF NOT EXISTS lead_scores (
            id SERIAL PRIMARY KEY,
            lead_id VARCHAR(255),
            score FLOAT,
            score_category VARCHAR(50),
            model_version VARCHAR(255),
            features_json TEXT,
            scored_at TIMESTAMP,
            FOREIGN KEY (lead_id) REFERENCES leads (lead_id)
        );
        """

        create_model_metadata = """
        CREATE TABLE IF NOT EXISTS model_metadata (
            id SERIAL PRIMARY KEY,
            model_version VARCHAR(255),
            algorithm VARCHAR(255),
            accuracy FLOAT,
            precision_score FLOAT,
            recall FLOAT,
            f1_score FLOAT,
            roc_auc FLOAT,
            trained_at TIMESTAMP,
            features_used TEXT
        );
        """

        with self.engine.begin() as conn:
            conn.execute(text(create_leads_table))
            conn.execute(text(create_scores_table))
            conn.execute(text(create_model_metadata))

    # ---------------------------------------------------------
    # Insert or update lead
    # ---------------------------------------------------------
    def insert_lead(self, lead_data: Dict):
        query = """
        INSERT INTO leads (
            lead_id, email, company_name, industry, job_title,
            seniority_level, company_size, country, created_at, updated_at
        )
        VALUES (
            :lead_id, :email, :company_name, :industry, :job_title,
            :seniority_level, :company_size, :country, :created_at, :updated_at
        )
        ON CONFLICT (lead_id) DO UPDATE SET
            email = EXCLUDED.email,
            company_name = EXCLUDED.company_name,
            industry = EXCLUDED.industry,
            job_title = EXCLUDED.job_title,
            seniority_level = EXCLUDED.seniority_level,
            company_size = EXCLUDED.company_size,
            country = EXCLUDED.country,
            updated_at = EXCLUDED.updated_at;
        """

        lead_data = lead_data.copy()
        lead_data["updated_at"] = datetime.now()
        lead_data.setdefault("created_at", datetime.now())

        with self.engine.begin() as conn:
            conn.execute(text(query), lead_data)

    # ---------------------------------------------------------
    # Insert score
    # ---------------------------------------------------------
    def insert_score(
        self, lead_id: str, score: float, category: str,
        model_version: str, features: str
    ):
        query = """
        INSERT INTO lead_scores (
            lead_id, score, score_category, model_version,
            features_json, scored_at
        )
        VALUES (
            :lead_id, :score, :score_category, :model_version,
            :features_json, :scored_at
        );
        """

        data = {
            "lead_id": lead_id,
            "score": score,
            "score_category": category,
            "model_version": model_version,
            "features_json": features,
            "scored_at": datetime.now()
        }

        with self.engine.begin() as conn:
            conn.execute(text(query), data)

    # ---------------------------------------------------------
    # Combined storage (used by API background task)
    # ---------------------------------------------------------
    def save_lead_and_score(self, lead_data: Dict, score_result: Dict):
        try:
            self.insert_lead(lead_data)
            self.insert_score(
                lead_id=score_result["lead_id"],
                score=score_result["lead_score"],
                category=score_result["score_category"],
                model_version=score_result["model_version"],
                features=str(lead_data),
            )
        except SQLAlchemyError as e:
            print(f"Error saving lead + score: {e}")

    # ---------------------------------------------------------
    # Batch storage
    # ---------------------------------------------------------
    def save_batch_scores(self, leads: List[Dict], scores: List[Dict]):
        for lead, score in zip(leads, scores):
            self.save_lead_and_score(lead, score)

    # ---------------------------------------------------------
    # Fetch recent lead scores
    # ---------------------------------------------------------
    def get_recent_scores(self, limit: int = 100) -> pd.DataFrame:
        query = f"""
        SELECT ls.*, l.email, l.company_name, l.industry
        FROM lead_scores ls
        JOIN leads l ON ls.lead_id = l.lead_id
        ORDER BY scored_at DESC
        LIMIT {limit};
        """

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    # ---------------------------------------------------------
    # Score distribution for dashboard
    # ---------------------------------------------------------
    def get_score_distribution(self) -> pd.DataFrame:
        query = """
        SELECT
            score_category,
            COUNT(*) as count,
            AVG(score) as avg_score,
            MIN(score) as min_score,
            MAX(score) as max_score
        FROM lead_scores
        WHERE scored_at >= NOW() - INTERVAL '30 days'
        GROUP BY score_category
        ORDER BY avg_score DESC;
        """

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)

    # ---------------------------------------------------------
    # Model performance logs for dashboard
    # ---------------------------------------------------------
    def get_model_performance(self) -> pd.DataFrame:
        query = """
        SELECT * FROM model_metadata
        ORDER BY trained_at DESC
        LIMIT 20;
        """

        with self.engine.connect() as conn:
            return pd.read_sql(query, conn)
