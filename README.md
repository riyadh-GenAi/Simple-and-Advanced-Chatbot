# Conversational Memory with LangChain & Groq

## Overview

This project demonstrates how to build a **stateful conversational chatbot** using **LangChain** and **Groq LLMs**, with a strong focus on **chat memory management**.

It shows:

* Stateless vs stateful conversations
* Session-based memory isolation
* Unlimited vs limited memory strategies
* How conversational context affects LLM responses

This is a **foundational pattern for production chatbots**, assistants, and multi-user AI systems.

---

## Key Features

* 🔹 Groq LLM integration (LLaMA 3.1 – 8B Instant)
* 🔹 Deterministic responses (`temperature=0`)
* 🔹 Stateless chatbot invocation
* 🔹 Session-based conversational memory
* 🔹 Multi-user isolation via `session_id`
* 🔹 Memory windowing (limited context)
* 🔹 Clean separation of prompt, memory, and model

---

## Tech Stack

* **Python 3.10+**
* **LangChain Core & Community**
* **Groq (LLaMA 3.1 – 8B Instant)**
* **python-dotenv**

---

## High-Level Architecture

```
User Input
   ↓
RunnableWithMessageHistory
   ↓
ChatMessageHistory (per session)
   ↓
Prompt Template
   ↓
Groq LLM
   ↓
Response
```

---

## Project Flow Explanation

### 1. Environment & Model Setup

* Loads `GROQ_API_KEY` from `.env`
* Initializes Groq chat model with deterministic output

---

### 2. Stateless Chat Example

The first part of the code demonstrates **stateless invocation**:

```python
chatbot.invoke([HumanMessage(content="My Code Name is Riyadh_genAI")])
```

Each call is independent. The model **does not remember** previous messages unless they are explicitly re-sent.

---

### 3. Introducing Chat Memory

A dictionary-based in-memory store is used:

```python
chatbotMemory = {}
```

Each `session_id` maps to its own `ChatMessageHistory`, enabling:

* Multi-user support
* Isolated conversation states

---

### 4. Session-Based Conversations

```python
session1 = {"configurable": {"session_id": "001"}}
session2 = {"configurable": {"session_id": "002"}}
```

**Key behavior:**

* Session 001 remembers its own history
* Session 002 starts with a clean context

This mirrors **real production chat systems**.

---

### 5. RunnableWithMessageHistory

```python
RunnableWithMessageHistory(
    chatbot,
    get_session_history
)
```

This wrapper:

* Automatically appends user & AI messages
* Restores context on each request
* Keeps memory logic separate from business logic

---

## Unlimited vs Limited Memory

### Unlimited Memory

* Stores **entire conversation history**
* Best for short conversations
* Can grow expensive and slow

---

### Limited Memory (Sliding Window)

A custom function restricts context size:

```python
def limited_memory_of_messages(messages, number_of_messages_to_keep=2):
    return messages[-number_of_messages_to_keep:]
```

Only the **last N messages** are sent to the LLM.

---

### Why This Matters

| Memory Type | Pros                | Cons                |
| ----------- | ------------------- | ------------------- |
| Unlimited   | Full context        | Token growth, cost  |
| Limited     | Efficient, scalable | Partial memory loss |

Production systems often combine:

* Windowed memory
* Summarization memory
* Vector-based long-term memory

---

## Example Behaviors

### Session Isolation

* Session 001 remembers:

  * Code name
  * Preferences

* Session 002 does **not** know Session 001 data

---

### Memory Window Impact

* Limited-memory chatbot may forget early facts
* Unlimited-memory chatbot retains full history

This is intentional and configurable.

---

## Folder Structure (Suggested)

```
project-root/
│── main.py
│── .env
│── .gitignore
│── requirements.txt
│── README.md
```

---

## Environment Setup

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Required Dependencies

```txt
langchain
langchain-groq
langchain-core
langchain-community
python-dotenv
```

---

## Practice vs Production Comparison

| Aspect       | Demo           | Production      |
| ------------ | -------------- | --------------- |
| Memory Store | In-memory dict | Redis / DB      |
| Sessions     | Hardcoded      | Auth-based      |
| Scaling      | Single process | Distributed     |
| Cost Control | None           | Token budgeting |
| Persistence  | Volatile       | Durable         |

---

## Production Extensions

* Redis-backed message history
* User authentication
* Memory summarization
* Vector-based long-term memory
* Observability & logging

---

## Common Use Cases

* Chatbots & virtual assistants
* Customer support systems
* AI tutors
* Multi-user LLM applications
* Conversational agents with context

---

## Author

**Riyadh GenAI**
Generative AI & LLM Systems Engineering

---


