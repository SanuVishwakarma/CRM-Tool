"""
CrewAI Agent Definitions using Groq LLM for Lead Score Flow
"""

import yaml
from crewai import Agent
from pathlib import Path

from src.llm.groq_llm import GroqLLM


class LeadScoringAgents:
    """
    Defines all agents used in the Lead Score Flow pipeline.
    Each agent uses a dedicated GroqLLM model per config settings.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load model definitions from config
        self.agent_models = self.config["llm"]["agents"]
        self.verbose = self.config["crew"]["verbose"]

    # -----------------------------------------------------------
    # Helper: Build GroqLLM for a given agent type
    # -----------------------------------------------------------
    def _build_llm(self, agent_key: str) -> GroqLLM:
        model_name = self.agent_models[agent_key]["model"]
        temp = self.agent_models[agent_key]["temperature"]

        return GroqLLM(model=model_name, temperature=temp)

    # -----------------------------------------------------------
    # 1. Data Analyst Agent
    # -----------------------------------------------------------
    def data_analyst_agent(self) -> Agent:
        return Agent(
            role="Lead Data Analyst",
            goal=(
                "Analyze the dataset, identify important behavioral patterns, "
                "highlight anomalies, and suggest feature engineering improvements."
            ),
            backstory=(
                "You are an expert data analyst specializing in customer engagement "
                "and lead behavior. You excel at identifying patterns in large datasets "
                "and translating them into actionable insights."
            ),
            verbose=self.verbose,
            allow_delegation=False,
            llm=self._build_llm("data_analyst"),
        )

    # -----------------------------------------------------------
    # 2. Machine Learning Engineer Agent
    # -----------------------------------------------------------
    def ml_engineer_agent(self) -> Agent:
        return Agent(
            role="Machine Learning Engineer",
            goal=(
                "Evaluate model performance metrics, analyze feature importance, and "
                "recommend improvements to increase predictive accuracy."
            ),
            backstory=(
                "You are a senior ML engineer experienced with classification, "
                "imbalanced datasets, model tuning, and evaluating production ML systems."
            ),
            verbose=self.verbose,
            allow_delegation=False,
            llm=self._build_llm("ml_engineer"),
        )

    # -----------------------------------------------------------
    # 3. Lead Scoring Specialist Agent
    # -----------------------------------------------------------
    def scoring_specialist_agent(self) -> Agent:
        return Agent(
            role="Lead Scoring Specialist",
            goal=(
                "Evaluate lead scoring outcomes, validate scoring calibration, "
                "and ensure alignment with business expectations."
            ),
            backstory=(
                "You are an expert in interpreting scoring models and business rules. "
                "You ensure leads are categorized effectively for sales prioritization."
            ),
            verbose=self.verbose,
            allow_delegation=False,
            llm=self._build_llm("scoring_specialist"),
        )

    # -----------------------------------------------------------
    # 4. Business Intelligence Analyst Agent
    # -----------------------------------------------------------
    def bi_analyst_agent(self) -> Agent:
        return Agent(
            role="Business Intelligence Analyst",
            goal=(
                "Generate clear, executive-level insights from scoring results and model "
                "performance. Produce actionable recommendations."
            ),
            backstory=(
                "You are a highly skilled BI analyst known for summarizing complex data "
                "into simple, impactful business narratives for decision-makers."
            ),
            verbose=self.verbose,
            allow_delegation=False,
            llm=self._build_llm("bi_analyst"),
        )
