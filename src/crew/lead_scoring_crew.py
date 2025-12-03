"""
CrewAI Orchestration for Lead Score Flow using Groq LLMs.
Handles multi-agent workflows: data analysis, model evaluation,
scoring analysis, and executive reporting.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from crewai import Crew, Process

from src.agents.crew_agents import LeadScoringAgents
from src.tasks.task_definitions import LeadScoringTasks


class LeadScoringCrew:
    """
    Orchestrates all CrewAI agent workflows in the Lead Score Flow system.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize agent & task managers
        self.agents = LeadScoringAgents(config_path=config_path)
        self.tasks = LeadScoringTasks()

        # Agent objects
        self.data_analyst = self.agents.data_analyst_agent()
        self.ml_engineer = self.agents.ml_engineer_agent()
        self.scoring_specialist = self.agents.scoring_specialist_agent()
        self.bi_analyst = self.agents.bi_analyst_agent()

        self.verbose = self.config["crew"]["verbose"]

    # ========================================================================
    # SUPPORTING CONTEXT BUILDERS
    # ========================================================================

    def create_data_summary(
        self, df: pd.DataFrame, has_labels: bool = False
    ) -> Dict[str, Any]:
        """
        Create dataset summary for the Data Analyst Agent.
        """

        summary = {
            "total_rows": len(df),
            "num_features": len(df.columns),
            "feature_list": df.columns.tolist(),
        }

        if "created_at" in df.columns:
            summary["date_range"] = f"{df['created_at'].min()} to {df['created_at'].max()}"

        if has_labels and "converted" in df.columns:
            summary.update(
                {
                    "conversion_rate": float(df["converted"].mean()),
                    "converted_total": int(df["converted"].sum()),
                    "not_converted_total": int((1 - df["converted"]).sum()),
                }
            )

        return summary

    def create_model_info(
        self, metrics: Dict[str, Any], train_size: int, test_size: int
    ) -> Dict[str, Any]:
        """
        Create ML model information summary.
        """

        return {
            "algorithm": metrics.get("algorithm"),
            "train_samples": train_size,
            "test_samples": test_size,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "roc_auc": metrics.get("roc_auc"),
            "features_used": metrics.get("features_used"),
        }

    def create_scoring_info(
        self, total_leads: int, model_version: str, method: str
    ) -> Dict[str, Any]:
        """
        Create scoring context for the Scoring Specialist agent.
        """

        return {
            "total_leads": total_leads,
            "model_version": model_version,
            "method": method,
            "score_ranges": self.config["model"]["score_ranges"],
        }

    def create_report_data(
        self, insights: Dict[str, Any], metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide information for BI Report Agent.
        """

        return {
            "total_leads": insights.get("total_leads"),
            "hot_leads": insights.get("hot_leads"),
            "warm_leads": insights.get("warm_leads"),
            "cold_leads": insights.get("cold_leads"),
            "avg_score": float(insights.get("avg_score", 0)),
            "median_score": float(insights.get("median_score", 0)),
            "model_accuracy": float(metrics.get("accuracy", 0)),
            "category_distribution": insights.get("category_distribution"),
        }

    # ========================================================================
    # WORKFLOWS
    # ========================================================================

    # ---------------------------------------------------------
    # 1. Data Analysis Workflow
    # ---------------------------------------------------------
    def run_analysis_workflow(self, df: pd.DataFrame, has_labels: bool = False) -> str:
        print("\n🚀 Starting Data Analysis Workflow...")

        summary = self.create_data_summary(df, has_labels)

        task = self.tasks.data_analysis_task(self.data_analyst, summary)

        crew = Crew(
            agents=[self.data_analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        print("✅ Data Analysis Completed!")
        return result

    # ---------------------------------------------------------
    # 2. Model Evaluation Workflow
    # ---------------------------------------------------------
    def run_modeling_workflow(
        self, metrics: Dict[str, Any], train_size: int, test_size: int
    ) -> str:
        print("\n🚀 Starting Model Evaluation Workflow...")

        model_info = self.create_model_info(metrics, train_size, test_size)

        task = self.tasks.model_training_task(self.ml_engineer, model_info)

        crew = Crew(
            agents=[self.ml_engineer],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        print("✅ Model Evaluation Completed!")
        return result

    # ---------------------------------------------------------
    # 3. Lead Scoring Workflow
    # ---------------------------------------------------------
    def run_scoring_workflow(
        self, total_leads: int, model_version: str, method: str
    ) -> str:
        print("\n🚀 Starting Lead Scoring Workflow...")

        scoring_info = self.create_scoring_info(
            total_leads=total_leads, model_version=model_version, method=method
        )

        task = self.tasks.lead_scoring_task(self.scoring_specialist, scoring_info)

        crew = Crew(
            agents=[self.scoring_specialist],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        print("✅ Lead Scoring Workflow Completed!")
        return result

    # ---------------------------------------------------------
    # 4. Executive Reporting Workflow
    # ---------------------------------------------------------
    def run_reporting_workflow(
        self, insights: Dict[str, Any], metrics: Dict[str, Any]
    ) -> str:
        print("\n🚀 Starting Executive Reporting Workflow...")

        report_data = self.create_report_data(insights, metrics)

        task = self.tasks.reporting_task(self.bi_analyst, report_data)

        crew = Crew(
            agents=[self.bi_analyst],
            tasks=[task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        print("✅ Executive Reporting Completed!")
        return result
