import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import streamlit as st
import google.generativeai as genai


# ENV
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")


# Configure API key
genai.configure(api_key=GOOGLE_API_KEY)

# LLM
llm = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0
    }
)


# Streamlit UI
st.set_page_config(page_title="Postgres SQL Chatbot")
st.title("🎵 Chat with Chinook DB")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Database connection
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

# Get database schema as a formatted string for the LLM prompt
@st.cache_data
def get_schema():

    engine = get_engine()
    inspector_query = text("""
SELECT table_name, column_name
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
""")

    schema_str = ""

    try:
        with engine.connect() as conn:
            results = conn.execute(inspector_query)
            current_table = ""
            for row in results:
                table_name, column_name = row[0],row[1]
                if table_name != current_table:
                    schema_str += f"\nTable: {table_name} \nColumns: "
                    current_table = table_name
                schema_str += f"{column_name}, "
    except Exception as e:
        st.error(f"Error reading schema: {e}")

    return schema_str


# Generate SQL query from user question and database schema using the LLM
def get_sql_from_openai(question, schema):
    prompt = f"""
You are a PostgreSQL expert data analyst.

Database schema:
{schema}

User question:
{question}

Task:
Write a PostgreSQL SELECT query to answer the question.

CRITICAL RULES:

1. Output ONLY SQL — no explanations, no markdown.
2. Use DISTINCT when removing duplicates.
3. Tables were created via Pandas → identifiers are case-sensitive.
4. ALWAYS wrap table names in double quotes.
   Example: "Album", "Track", "Genre"
5. ALWAYS wrap column names in double quotes — even when using alias dot notation.
   Example: A."Name", T."Title", G."Name"
   NEVER write: A.Name, T.Title, G.Name  ← THIS IS WRONG
6. DO NOT quote aliases.
   Example: "Album" AS A  ✅
            "Album" AS "A" ❌
7. Use clear aliases:
   Album → A
   Track → T
   Genre → G
8. Use correct foreign keys for joins.
9. Ensure the query runs on PostgreSQL.

COLUMN REFERENCE FORMAT:
- With alias:    alias."ColumnName"   ✅   e.g. A."Name", T."UnitPrice"
- Without alias: "TableName"."ColumnName"  ✅  e.g. "Artist"."Name"
- NEVER use unquoted column names in any context.

Return ONLY the SQL query.
"""

    response = llm.generate_content(prompt)
    clean_sql = (
        response.text
        .replace("```sql", "")
        .replace("```", "")
        .replace("'''sql", "")
        .replace("'''", "")
        .strip()
    )

    return clean_sql

# Run SQL query and return results as a DataFrame
def run_query(query):
    engine = get_engine()
    with engine.connect() as conn:
        try:
            result = conn.execute(text(query))
            return pd.DataFrame(result.fetchall(),columns=result.keys())
        
        except Exception as e:
            return str(e)
        

# Convert SQL query results into a natural language answer
def get_natural_response(question, sql, data):
    prompt = f"""
User question: {question}
SQL query used: {sql}
Data retrived: {data}

Task: 
Answer the user's question in a natural language format based on the SQL query data
If the data is empty, say "No results found".
"""
    
    response = llm.generate_content(prompt)
    return response.text


# Display previous chat messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        if msg["type"] == "text":
            st.write(msg["content"])

        elif msg["type"] == "sql":
            st.code(msg["content"], language="sql")

        elif msg["type"] == "data":
            st.dataframe(msg["content"])


# Streamlit UI
question = st.chat_input("Ask a question about the Chinook database")

if question:

    # user message
    st.session_state.messages.append({
        "role": "user",
        "type": "text",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)


    schema = get_schema()

    sql_query = get_sql_from_openai(question, schema)

    data = run_query(sql_query)

    answer = get_natural_response(question, sql_query, data)


    # Store messages
    st.session_state.messages.append({
        "role": "assistant",
        "type": "sql",
        "content": sql_query
    })

    st.session_state.messages.append({
        "role": "assistant",
        "type": "data",
        "content": data
    })

    st.session_state.messages.append({
        "role": "assistant",
        "type": "text",
        "content": answer
    })


    # Response
    with st.chat_message("assistant"):
        st.markdown("**Generated SQL**")
        st.code(sql_query, language="sql")

        st.markdown("**Query Result**")
        st.dataframe(data)

        st.markdown("**Answer**")
        st.write(answer)