"""
FastAPI Application for Lead Score Flow (Groq + PostgreSQL Version)
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_engineering import FeatureEngineer
from models.scorer import LeadScorer
from database.db_manager import DatabaseManager


# ---------------------------------------------
# Initialize API
# ---------------------------------------------
app = FastAPI(
    title="Lead Score Flow API",
    description="AI-Powered Lead Scoring System using Groq LLM + ML Models",
    version="2.0.0",
)

# Initialize core components
feature_engineer = FeatureEngineer()
scorer = LeadScorer()
db = DatabaseManager()


# ---------------------------------------------
# Pydantic Input Models
# ---------------------------------------------
class LeadInput(BaseModel):
    lead_id: str

    email: Optional[str] = None
    company_name: Optional[str] = None

    industry: str
    job_title: str
    seniority_level: str
    company_size: str
    country: str

    email_opens: int = Field(ge=0)
    email_clicks: int = Field(ge=0)
    website_visits: int = Field(ge=0)
    time_on_site: float = Field(ge=0)
    pages_viewed: int = Field(ge=0)
    content_downloads: int = Field(ge=0)
    form_submissions: int = Field(ge=0)
    event_attendance: int = Field(ge=0)
    demo_requests: int = Field(ge=0)
    pricing_page_views: int = Field(ge=0)
    feature_page_views: int = Field(ge=0)

    days_since_first_interaction: int = Field(ge=0)
    total_interactions: int = Field(ge=0)
    last_interaction_days: int = Field(ge=0)


class LeadScoreResponse(BaseModel):
    lead_id: str
    lead_score: float
    score_category: str
    score_rank: Optional[int] = None
    model_version: str
    scored_at: datetime


class BatchLeadInput(BaseModel):
    leads: List[LeadInput]


# ---------------------------------------------
# Root Endpoint
# ---------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Lead Score Flow API (Groq + ML)",
        "version": "2.0.0",
        "endpoints": {
            "score_single": "/api/score",
            "score_batch": "/api/score/batch",
            "stats": "/api/stats",
            "top_leads": "/api/top-leads",
            "health": "/health",
        },
    }


# ---------------------------------------------
# Health Check
# ---------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": scorer.model_version,
        "database": "connected" if db.test_connection() else "error",
    }


# ---------------------------------------------
# Score Single Lead
# ---------------------------------------------
@app.post("/api/score", response_model=LeadScoreResponse)
async def score_lead(lead: LeadInput, background_tasks: BackgroundTasks):

    try:
        df = pd.DataFrame([lead.dict()])

        # Add interaction frequency
        df["interaction_frequency"] = df["total_interactions"] / (
            df["days_since_first_interaction"] + 1
        )

        # Prepare features
        df_processed, feature_cols = feature_engineer.prepare_data(
            df, fit=False, scale=True
        )
        X = df_processed[feature_cols]

        # Score lead
        scored_df = scorer.score_leads(df_processed, X, method="hybrid")
        row = scored_df.iloc[0]

        result = {
            "lead_id": lead.lead_id,
            "lead_score": float(row["lead_score"]),
            "score_category": row["score_category"],
            "model_version": scorer.model_version,
            "scored_at": datetime.now(),
        }

        # Background DB save
        background_tasks.add_task(db.save_lead_and_score, lead.dict(), result)

        return LeadScoreResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring Error: {e}")


# ---------------------------------------------
# Batch Scoring
# ---------------------------------------------
@app.post("/api/score/batch")
async def score_batch(input_data: BatchLeadInput, background_tasks: BackgroundTasks):

    try:
        leads_data = [l.dict() for l in input_data.leads]
        df = pd.DataFrame(leads_data)

        df["interaction_frequency"] = df["total_interactions"] / (
            df["days_since_first_interaction"] + 1
        )

        df_processed, feature_cols = feature_engineer.prepare_data(
            df, fit=False, scale=True
        )
        X = df_processed[feature_cols]

        scored_df = scorer.score_leads(df_processed, X, method="hybrid")

        results = []
        for _, row in scored_df.iterrows():
            results.append(
                {
                    "lead_id": row["lead_id"],
                    "lead_score": float(row["lead_score"]),
                    "score_category": row["score_category"],
                    "score_rank": int(row["score_rank"]),
                    "model_version": row["model_version"],
                    "scored_at": datetime.now(),
                }
            )

        background_tasks.add_task(db.save_batch_scores, leads_data, results)

        return {
            "total_leads": len(results),
            "scores": results,
            "summary": {
                "avg_score": float(scored_df["lead_score"].mean()),
                "hot": int((scored_df["score_category"] == "hot").sum()),
                "warm": int((scored_df["score_category"] == "warm").sum()),
                "cold": int((scored_df["score_category"] == "cold").sum()),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch Scoring Error: {e}")


# ---------------------------------------------
# Scoring Statistics
# ---------------------------------------------
@app.get("/api/stats")
async def get_stats():
    try:
        distribution = db.get_score_distribution()
        recent = db.get_recent_scores(limit=100)

        return {
            "total_scored": len(recent),
            "score_distribution": (
                distribution.to_dict("records") if not distribution.empty else []
            ),
            "avg_score_last_100": (
                float(recent["score"].mean()) if not recent.empty else 0
            ),
            "model_version": scorer.model_version,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats Error: {e}")


# ---------------------------------------------
# Top Leads
# ---------------------------------------------
@app.get("/api/top-leads")
async def top_leads(limit: int = 10, category: Optional[str] = None):

    try:
        recent = db.get_recent_scores(limit=1000)

        if category:
            recent = recent[recent["score_category"] == category]

        top = recent.nlargest(limit, "score")

        return {
            "count": len(top),
            "top_leads": top.to_dict("records"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Top Leads Error: {e}")
