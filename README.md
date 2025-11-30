# 🎯 Customer Relationship Management System

**AI-Powered Lead Scoring System using CrewAI, Python & Machine Learning**

A complete, production-ready lead scoring system that uses agentic AI (CrewAI) to automate the entire workflow from data analysis to actionable insights.

## ✨ Features

- **🤖 Agentic AI Workflow**: CrewAI agents handle data analysis, model training, scoring, and reporting
- **📊 Advanced ML Pipeline**: XGBoost, Random Forest, and hybrid scoring approaches
- **⚡ Real-time API**: FastAPI endpoint for instant lead scoring
- **📈 Interactive Dashboard**: Streamlit-based insights and visualization
- **💾 Database Integration**: SQLite/PostgreSQL for persistent storage
- **🔄 Automated Workflow**: End-to-end pipeline from ingestion to insights

## 🏗️ Architecture

```
Lead Data → Feature Engineering → ML Model Training → Lead Scoring → Insights
     ↓              ↓                     ↓                ↓            ↓
   CrewAI      Data Analyst         ML Engineer      Scoring        BI Analyst
   Agents        Agent                Agent           Agent           Agent
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Groq API Key (for CrewAI agents)

### Installation

1. **Clone and Setup**
```bash
git clone <repository>
cd lead-score-flow
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. **Generate Sample Data**
```bash
python scripts/generate_sample_data.py
```

4. **Run Complete Pipeline**
```bash
python main.py
```

This will:
- Load and analyze data (CrewAI Data Analyst)
- Train ML model (CrewAI ML Engineer)
- Score all leads (CrewAI Scoring Specialist)
- Generate insights (CrewAI BI Analyst)
- Store results in database

## 📊 Using the System

### 1. API Server

Start the FastAPI server:
```bash
python -m uvicorn src.api.main:app --reload
```

Access API documentation: `http://localhost:8000/docs`


### 2. Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run src/dashboard/app.py
```

Access dashboard: `http://localhost:8501`

**Dashboard Features:**
- Real-time lead scoring metrics
- Score distribution visualizations
- Top leads identification
- Model performance tracking
- Industry analysis
- Exportable reports


## 📁 Project Structure

```
lead-score-flow/
├── src/
│   ├── agents/              # CrewAI agent definitions
│   ├── tasks/               # Task definitions
│   ├── crew/                # Crew orchestration
│   ├── models/              # ML models & scoring
│   ├── data/                # Data processing
│   ├── api/                 # FastAPI application
│   ├── database/            # Database operations
│   └── dashboard/           # Streamlit dashboard
├── data/
│   ├── raw/                 # Raw lead data
│   ├── processed/           # Processed features
│   └── scored/              # Scored leads
├── models/                  # Saved ML models
├── reports/                 # Agent reports
├── config/                  # Configuration files
├── scripts/                 # Utility scripts
├── main.py                  # Main pipeline
└── requirements.txt
```


## 📊 Scoring Methodology

### Hybrid Scoring Approach

The system uses a hybrid scoring method combining:

1. **ML-Based Score (70% weight)**
   - XGBoost classification model
   - Trained on historical conversion data
   - Probability score scaled to 0-100

2. **Rule-Based Score (30% weight)**
   - Demo requests (25 points)
   - Pricing page views (15 points)
   - Decision maker + recent activity (10 points)
   - Form submissions (8 points)
   - Executive + large company (12 points)

### Lead Categories

- **🔥 Hot (80-100)**: Immediate sales action required
- **♨️ Warm (60-79)**: High potential, nurture with targeted content
- **❄️ Cold (40-59)**: Long-term nurturing, educational content
- **🚫 Unqualified (0-39)**: Re-engage or disqualify



## 🔄 Workflow Automation

Run automated scoring on schedule:

```bash
# Using cron (Linux/Mac)
0 9 * * * cd /path/to/lead-score-flow && python main.py

# Using Task Scheduler (Windows)
# Create a task to run main.py daily
```


## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

🚀 Start scoring smarter, not harder!
