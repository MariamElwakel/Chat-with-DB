# 🎵 Chinook SQL Chatbot

A Streamlit-based chatbot that converts Natural Language into SQL queries to explore the Chinook PostgreSQL database.  
The application connects to PostgreSQL using SQLAlchemy and executes dynamically generated queries.

This project includes three implementations:

1. **Without LangChain** → Direct LLM prompting (Google Gemini)
2. **With LangChain** → Structured chains, intent detection, tracing (OpenAI)
3. **Agent + Few‑Shots (RAG)** → SQL Agent enhanced with semantic few‑shot retrieval

---

# 📂 Files

- app_basic.py → Uses LLM + manual prompting  
- app_langchain.py → Uses LangChain + OpenAI + intent routing  

### 📁 agent/

This folder contains the third approach:

- app_agent.py → SQL Agent with tool calling + retrieval augmentation  
- fewshots.json → Curated question‑SQL examples used for semantic few‑shot retrieval  

The agent retrieves similar examples and injects them into the prompt to improve SQL generation accuracy.

---

# 🧪 Setup

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 2️⃣ Activate venv

```bash
venv\Scripts\activate
```

---

## 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file:

```env
DB_URL=postgresql://username:password@localhost:5432/Chinook

# For Gemini app
GOOGLE_API_KEY=your_google_key

# LangChain & Agent apps
OPENAI_API_KEY=your_openai_key
LANGSMITH_API_KEY=your_langsmith_key
```

---

# ▶️ How to Run

## Run without LangChain

```bash
streamlit run app_basic.py
```

## Run with LangChain

```bash
streamlit run app_langchain.py
```

## Run Agent + Few‑Shots (RAG)

```bash
streamlit run agent/app_agent.py
```

---

# 🧠 How It Works

## 1️⃣ Without LangChain

Pipeline:

1. Extract database schema  
2. LLM generates SQL  
3. Execute query via SQLAlchemy  
4. LLM converts results to natural language  
5. Display in Streamlit chat UI  

---

## 2️⃣ With LangChain

Adds orchestration layers:

- **Intent Detection (chat vs db)**  
  Detects whether the user input is:
  - Natural conversation (greetings, thanks, small talk) → handled as chat  
  - Data-related questions → routed to SQL generation  

- SQL generation chain  
- Natural language response chain  
- LangSmith tracing & observability  
- Structured prompt templates  

This separation prevents unnecessary SQL generation for casual conversation while ensuring database questions are answered accurately.

---

## 3️⃣ Agent + Few‑Shots (RAG)

Enhances the LangChain approach with semantic retrieval.

Flow:

1. User asks a question  
2. Similar question‑SQL pairs are retrieved from `fewshots.json`  
3. Retrieved examples are injected into the agent prompt  
4. SQL Agent generates and executes the query  
5. Returns a grounded natural language answer  

### Benefits

- Higher SQL syntax accuracy  
- Better aggregation handling  
- Improved multi‑table joins  
- Country normalization (e.g., USA vs United States)  
- Date casting & year extraction guidance  
- Reduced hallucinations  

---
