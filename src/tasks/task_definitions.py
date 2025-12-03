"""
Task definitions for CrewAI Agents in the Lead Score Flow system.
Each task provides structured instructions to guide Groq-powered agents.
"""

from crewai import Task


class LeadScoringTasks:
    """
    Defines all task templates used in the CrewAI workflow.
    """

    # ---------------------------------------------------------
    # 1. Data Analysis Task
    # ---------------------------------------------------------
    def data_analysis_task(self, agent, data_summary: dict) -> Task:
        """
        Provide dataset overview to the Data Analyst Agent.
        """

        prompt = f"""
You are the Lead Data Analyst. Analyze the dataset summary below and produce:

1. Key behavioral patterns and correlations.
2. Potential outliers or anomalies.
3. Recommendations for feature engineering.
4. What signals most strongly indicate lead conversion.

DATA SUMMARY:
{data_summary}

IMPORTANT:
- Write clearly and concisely.
- Provide actionable, data-driven insights.
"""

        return Task(
            description="Analyze dataset structure and identify ML-relevant patterns.",
            expected_output=(
                "Detailed analysis including feature insights, anomalies, and "
                "recommendations for model improvement."
            ),
            agent=agent,
            prompt=prompt.strip()
        )

    # ---------------------------------------------------------
    # 2. Model Training / Evaluation Task
    # ---------------------------------------------------------
    def model_training_task(self, agent, model_info: dict) -> Task:
        """
        Provide model performance metrics to ML engineer agent.
        """

        prompt = f"""
You are the Machine Learning Engineer. Evaluate the following model metrics and produce:

1. Strengths and weaknesses of the model.
2. Issues related to precision, recall, or calibration.
3. Insights from feature importance.
4. Recommendations to improve accuracy and reduce error.

MODEL INFO:
{model_info}

IMPORTANT:
- Make insights ML-focused.
- Provide technical and practical improvement suggestions.
"""

        return Task(
            description="Evaluate ML model performance and suggest improvements.",
            expected_output=(
                "A detailed ML engineering evaluation including strengths, weaknesses, "
                "and actionable recommendations."
            ),
            agent=agent,
            prompt=prompt.strip()
        )

    # ---------------------------------------------------------
    # 3. Lead Scoring Task
    # ---------------------------------------------------------
    def lead_scoring_task(self, agent, scoring_info: dict) -> Task:
        """
        Evaluate scoring methodology and categories.
        """

        prompt = f"""
You are the Lead Scoring Specialist. Review the scoring process:

SCORING INFO:
{scoring_info}

Your responsibilities:
1. Validate that the scoring method (ML + rules) is balanced.
2. Identify any scoring bias or misalignment.
3. Suggest improvements to category thresholds.
4. Ensure categories reflect business priorities.

IMPORTANT:
- Use a hybrid ML + rule-based mindset.
- Provide improvements to scoring calibration.
"""

        return Task(
            description="Evaluate scoring methodology for correctness and improvements.",
            expected_output=(
                "A comprehensive scoring evaluation including calibration checks and "
                "business-aligned recommendations."
            ),
            agent=agent,
            prompt=prompt.strip()
        )

    # ---------------------------------------------------------
    # 4. BI Reporting Task
    # ---------------------------------------------------------
    def reporting_task(self, agent, report_data: dict) -> Task:
        """
        Produce an executive-level analysis of insights and metrics.
        """

        prompt = f"""
You are the Business Intelligence Analyst. Using the insights and data provided:

REPORT DATA:
{report_data}

Produce a polished executive summary including:
1. High-level narrative of lead behavior and scoring outcomes.
2. Key insights from score distribution.
3. Performance interpretation from ML metrics.
4. Business recommendations for sales and marketing.
5. Risks, opportunities, and next steps.

IMPORTANT:
- Write in clear business English.
- Insights must be actionable, not technical.
- Keep formatting structured for executives.
"""

        return Task(
            description="Generate a high-quality executive BI report.",
            expected_output=(
                "A well-written executive summary containing insights and "
                "business recommendations."
            ),
            agent=agent,
            prompt=prompt.strip()
        )
