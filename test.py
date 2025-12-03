# import psycopg2
# hostname = 'localhost'
# database = 'lead_scoring'
# username = 'postgres'
# password = 7268
# port_id = 5432

# conn = psycopg2.connect(
#     host=hostname,
#     dbname=database,
#     user=username,
#     password=password,
#     port=port_id
# )

# # conn.close()
# import pandas as pd
# from src.database.db_manager import DatabaseManager
# db = DatabaseManager()
# print("DB OK:", db.test_connection())

from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Hello from Groq!"}]
)

print(response.choices[0].message.content)





