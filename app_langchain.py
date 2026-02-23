import os
import pandas as pd
from dotenv import load_dotenv
import streamlit as st

from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langsmith import Client


# ENV
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DB_URL")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Chinook SQL Chatbot with LangChain"
Client()

# LLM
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=OPENAI_API_KEY,
    temperature=0
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
    SELECT
        table_name,
        column_name,
        data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;
    """)

    schema_str = ""
    current_table = None

    try:
        with engine.connect() as conn:
            results = conn.execute(inspector_query)

            for table_name, column_name, data_type in results:

                # New table header
                if table_name != current_table:
                    schema_str += f'\nTable: "{table_name}"\nColumns:\n'
                    current_table = table_name

                # Column with type
                schema_str += (
                    f'- "{column_name}" ({data_type})\n'
                )

    except Exception as e:
        st.error(f"Error reading schema: {e}")

    return schema_str


# Prompt Template for normal chat (greetings, small talk, etc.)
chat_prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are a friendly conversational assistant for a database chat application.

Your role is to chat naturally with the user when their message does NOT require querying the database.

Behavior guidelines:

1. Greetings & small talk
   - Greet the user warmly.
   - Be friendly, polite, and approachable.
   - Example: “Hi there! How can I help you today?”

2. Offer help proactively
   - Mention that you can help explore the Chinook music database.
   - Encourage questions about artists, albums, tracks, customers, invoices, etc.
   - Example: “Feel free to ask me anything about the music store data.”

3. Stay concise
   - Keep responses short and conversational.
   - Avoid long explanations unless asked.

4. Do NOT generate or mention SQL
   - If the user is chatting casually, never talk about queries, schemas, or technical details.

5. Handle vague questions
   - If the user says something unclear like “help me” → guide them.
   - Example: “Sure — what would you like to know about the database?”

6. Handle appreciation
   - If user says “thanks” → respond politely.
   - Example: “You’re welcome! Let me know if you need anything else.”

7. Handle capability questions
   - If user asks what you can do → explain briefly.
   - Example: “I can help you explore artists, albums, sales, and more.”

8. Tone
   - Friendly, supportive, non-robotic
   - Professional but casual
   - Never overly formal

Important constraint:
If the user asks for specific data or information that exists in the database, you should NOT answer it yourself — that will be handled by the database query system.

You only handle normal conversation.
"""
),
("human", "{question}")
]
)

# Normal chat chain
chat_chain = chat_prompt | llm

# Get normal chat response
def normal_chat_response(question):

    response = chat_chain.invoke(
        {"question": question}
    )

    return response.content


# Prompt Template for a SQL Query Generator
sql_prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an expert PostgreSQL SQL generator for the Chinook database.

You will be given a database schema and a user question.

Your job is to generate SQL that retrieves the answer ONLY from the database.

STRICT RULES:

1. ONLY use tables and columns that exist in the schema.
2. NEVER fabricate knowledge.
3. NEVER answer using general world knowledge.
4. NEVER return text explanations.
5. NEVER use SELECT with hard-coded strings.
6. If the question cannot be answered from the schema, return EXACTLY: NOT_POSSIBLE.
7. ALWAYS wrap table and column names in double quotes.
8. Return ONLY SQL or NOT_POSSIBLE.

CHINOOK DATA NORMALIZATION RULES:

9. Some country names in Chinook are stored as abbreviations.

   You MUST map user input to the stored database values.

   Known mappings include:

   - "United States" → 'USA'
   - "United States of America" → 'USA'
   - "US" → 'USA'

"""
),
("human",
"""
Schema:
{schema}

Question:
{question}
""")
])

# SQL query generator chain
sql_chain = sql_prompt | llm

# Get SQL query from LLM based on user question and database schema
def get_sql_from_llm(question, schema):

    response = sql_chain.invoke(
        {
            "question": question,
            "schema": schema
        },
        config={"run_name": "SQL Generation"}
    )

    clean_sql = response.content.strip()
    clean_sql = clean_sql.replace("```sql", "").replace("```", "")

    return clean_sql

# Run SQL query and return results as a DataFrame
def run_query(query):

    engine = get_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))

            return pd.DataFrame(
                result.fetchall(),
                columns=result.keys()
            )

    except Exception as e:
        # If query is wrong → return empty dataframe
        return pd.DataFrame()


# Prompt Template for intent detection (chat vs db)
intent_prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are an intent classifier for a Chinook SQL database assistant.

Classify the user message into ONE of two categories:

chat → ONLY if the message is clearly:
    - greeting (hi, hello, good morning)
    - thanks or appreciation
    - asking what the assistant can do
    - vague help like "help"

db → EVERYTHING ELSE.

Important rules:
- Any factual question must be classified as db.
- Any "who", "what", "when", "where", "how many" question is db, except if it's clearly about the assistant's capabilities or vague help.
- Any question about celebrities, weather, news, or public figures is db.
- Even if the information might not exist in the database, classify it as db.

Only return one word:
chat
or
db
"""
),
("human", "{question}")
]
)

# Intent detection chain
intent_chain = intent_prompt | llm

# Detect intent of user question (chat vs db)
def detect_intent(question):

    response = intent_chain.invoke(
        {"question": question},
        config={"run_name": "Intent Detection"}
    )

    intent = response.content.strip().lower()

    if "chat" in intent:
        return "chat"
    if "db" in intent:
        return "db"


# Prompt Template for converting results of SQL query into natural language answer
natural_prompt = ChatPromptTemplate.from_messages(
[
(
"system",
"""
You are a data analyst assistant.

Convert SQL results into a natural language answer.

If data is empty → say: "No results found".
"""
),
(
"human",
"""
User Question:
{question}

SQL Query:
{sql}

Data Retrieved:
{data}
"""
)
])

# Natural chat chain
natural_chain = natural_prompt | llm

# Get natural chat response
def get_natural_response(question, sql, data):

    response = natural_chain.invoke(
        {
            "question": question,
            "sql": sql,
            "data": data
        },
        config={"run_name": "Natural Language Response"}
    )

    return response.content


# Display chat history
for msg in st.session_state.messages:

    # User message
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["question"])

    # Assistant message
    else:
        with st.chat_message("assistant"):

            if msg["sql"]:

                st.subheader("Generated SQL")
                st.code(msg["sql"], language="sql")

                st.subheader("Query Result")
                st.write(msg["data"])

                st.subheader("Answer")
                st.write(msg["answer"])

            else:
                st.markdown(msg["answer"])


# Streamlit UI
question = st.chat_input("Ask anything about Chinook DB...")

if question:

    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "question": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Detect intent
    intent = detect_intent(question)

    # Normal chat mode
    if intent == "chat":

        answer = normal_chat_response(question)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "sql": "",
                "data": "",
                "answer": answer
            }
        )

    # Database query mode
    else:

        with st.spinner("Thinking..."):

            schema = get_schema()
            sql_query = get_sql_from_llm(question, schema)
            data = run_query(sql_query)
            answer = get_natural_response(
                question,
                sql_query,
                data
            )

        with st.chat_message("assistant"):

            st.subheader("Generated SQL")
            st.code(sql_query, language="sql")

            st.subheader("Query Result")
            st.write(data)

            st.subheader("Answer")
            st.write(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "sql": sql_query,
                "data": data,
                "answer": answer
            }
        )