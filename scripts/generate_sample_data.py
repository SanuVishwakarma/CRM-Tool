"""
Generate sample datasets for Lead Score Flow
Creates:
- data/raw/historical_leads.csv
- data/raw/new_leads.csv
"""

import pandas as pd
import numpy as np
import os
from faker import Faker

fake = Faker()


def generate_leads(n=1000, include_conversion=False):
    industries = ["Technology", "Finance", "Retail", "Healthcare", "Education"]
    seniority = ["Entry", "Mid", "Senior", "Executive"]
    company_sizes = ["1-10", "11-50", "51-200", "201-1000", "1000+"]
    countries = ["USA", "UK", "Germany", "Canada", "India"]

    rows = []

    for i in range(n):
        d = {
            "lead_id": f"LEAD_{i+1}",
            "email": fake.email(),
            "company_name": fake.company(),
            "industry": np.random.choice(industries),
            "job_title": fake.job(),
            "seniority_level": np.random.choice(seniority),
            "company_size": np.random.choice(company_sizes),
            "country": np.random.choice(countries),
            "email_opens": np.random.randint(0, 20),
            "email_clicks": np.random.randint(0, 10),
            "website_visits": np.random.randint(0, 40),
            "time_on_site": np.random.randint(20, 800),
            "pages_viewed": np.random.randint(1, 20),
            "content_downloads": np.random.randint(0, 5),
            "form_submissions": np.random.randint(0, 3),
            "event_attendance": np.random.randint(0, 2),
            "demo_requests": np.random.randint(0, 2),
            "pricing_page_views": np.random.randint(0, 5),
            "feature_page_views": np.random.randint(0, 10),
            "days_since_first_interaction": np.random.randint(1, 120),
            "total_interactions": np.random.randint(5, 60),
            "last_interaction_days": np.random.randint(1, 30)
        }

        if include_conversion:
            d["converted"] = np.random.choice([0, 1], p=[0.85, 0.15])

        rows.append(d)

    return pd.DataFrame(rows)


def main():
    os.makedirs("data/raw", exist_ok=True)

    print("Generating historical dataset...")
    historical = generate_leads(1500, include_conversion=True)
    historical.to_csv("data/raw/historical_leads.csv", index=False)
    print("Saved → data/raw/historical_leads.csv")

    print("Generating new dataset...")
    new = generate_leads(400, include_conversion=False)
    new.to_csv("data/raw/new_leads.csv", index=False)
    print("Saved → data/raw/new_leads.csv")

    print("Dataset generation complete ✔")


if __name__ == "__main__":
    main()
