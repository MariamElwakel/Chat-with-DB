import os
import streamlit as st
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ENV
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DB_URL")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Chinook SQL Chatbot Agent"
Client()


# Streamlit UI
st.set_page_config(page_title="Chinook SQL Agent")
st.title("🎵 Chat with Chinook DB")

if "messages" not in st.session_state:
    st.session_state.messages = []

# LLM
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)


# Database connection
@st.cache_resource
def get_db():
    return SQLDatabase.from_uri(DB_URL)


# Embeddings and Vectorstore for Fewshot Retrieval
@st.cache_resource
def get_fewshot_vectorstore():

    with open("fewshots.json", "r") as f:
        fewshots = json.load(f)

    documents = []

    for item in fewshots:
        content = f"""
Question:
{item['question']}

SQL:
{item['sql']}
"""
        documents.append(
            Document(
                page_content=content,
                metadata={"question": item["question"]}
            )
        )

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory="fewshot_chroma"
    )

    return vectordb


# Retrieve relevant fewshot examples based on user question
def retrieve_fewshots(user_question, k=3):

    vectordb = get_fewshot_vectorstore()

    docs = vectordb.similarity_search(user_question, k=k)

    # Deduplicate by question text
    seen = set()
    unique_docs = []

    for doc in docs:
        q = doc.metadata["question"]
        if q not in seen:
            seen.add(q)
            unique_docs.append(doc)

    fewshot_text = "\n\n".join([doc.page_content for doc in unique_docs])

    return fewshot_text


# System prompt for the agent
system_prompt = """
You are an expert PostgreSQL SQL assistant working with the Chinook database.

STRICT RULES:

1. ALWAYS wrap table names and column names in double quotes.
   Example:
   SELECT "Name" FROM "Artist"

2. Always preserve exact casing as defined in the schema.

3. Some country names in Chinook are stored as abbreviations.

   You MUST map user input to the stored database values.

   Known mappings include:

   - "United States" → 'USA'
   - "United States of America" → 'USA'
   - "US" → 'USA'

4. Only use columns and tables that exist in the schema.
5. Generate syntactically correct PostgreSQL queries.
6. Use table aliases consistently.
7. Never guess column names.
8. Always return natural language answers, not SQL, to the user.
9. For any question requiring data retrieval, you MUST execute a SQL query using the available tools.
10. NEVER fabricate, estimate, or assume database results.
11. If the query returns a count, total, or numeric value, you MUST run the SQL and use the returned result.
12. Do NOT respond with placeholders such as:
   - "a certain number"
   - "some customers"
   - "various results"
13. Always base the final answer strictly on the SQL execution output.
"""


# Prompt for the agent that includes retrieved fewshot examples
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("system", "Here are similar example question-SQL pairs:\n{fewshots}"),
    ("human", "{question}")
])


# Create the agent with the SQLDatabaseToolkit
@st.cache_resource
def get_agent():
    db = get_db()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    return create_agent(
        llm,
        tools,
        system_prompt=system_prompt
    )


agent = get_agent()

# Prompt to handle normal chat messages that don't require database queries
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

# Chain for normal chat
chat_chain = chat_prompt | llm

# Get normal chat response
def normal_chat_response(question):

    response = chat_chain.invoke(
        {"question": question}
    )

    return response.content


# Prompt to detect intent (chat vs db)
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

# Intent detection Chain
intent_chain = intent_prompt | llm

# Detect intent
def detect_intent(question):

    response = intent_chain.invoke(
        {"question": question},
        config={"run_name": "Intent Detection"}
    )

    intent = response.content.strip().lower()

    if "chat" in intent:
        return "chat"

    # Default to db if uncertain
    return "db"


# Display chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat
question = st.chat_input("Ask anything about the Chinook database...")

if question:

    # Save user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                intent = detect_intent(question)

                if intent == "db":

                    fewshots = retrieve_fewshots(question, k=3)
                    print("\n========== RETRIEVED FEWSHOTS ==========\n")
                    print(fewshots)
                    print("\n========================================\n")

                    formatted_messages = rag_prompt.format_messages(
                        fewshots=fewshots,
                        question=question
                    )

                    final_answer = None

                    for step in agent.stream(
                        {"messages": formatted_messages},
                        stream_mode="values",
                    ):
                        step["messages"][-1].pretty_print()
                        final_answer = step["messages"][-1].content

                    answer = final_answer
                else:
                    answer = normal_chat_response(question)


            except Exception as e:
                answer = f"Error: {str(e)}"

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )