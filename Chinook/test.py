"""
Schema Inspection Utility

This script connects to the database using the DB_URL environment variable,
queries information_schema.columns to retrieve all tables and their columns
within the public schema, and prints a grouped, human-readable overview of
the database structure for quick reference and debugging.
"""


import os

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
 
# ENV
load_dotenv()
DB_URL = os.getenv("DB_URL")

# Database Connection 
engine = create_engine(DB_URL)

# Query to fetch table and column information
query = text("""
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
""")
 
 
# Function to fetch and print the schema
def fetch_schema():
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            current_table = ""
            for row in result:
                table_name, column_name = row[0], row[1]
                if table_name != current_table:
                    if current_table != "":
                        print()  
                    print(f"Table: {table_name}\nColumns: ", end="")
                    current_table = table_name
                print(f"{column_name}, ", end="")
            print()
    except Exception as e:
        print(f"Error fetching schema: {e}")
 
 
if __name__ == "__main__":
    fetch_schema()