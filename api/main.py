from dotenv import load_dotenv
load_dotenv()
import os
import sys
import json
import sqlite3
import joblib
import pandas as pd
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Path Setup (Must happen before importing our custom modules)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '../data/warehouse.db')

sys.path.append(os.path.join(BASE_DIR, '..'))
from rag.retriever import retrieve

# 2. App Initialization
app = FastAPI(title="Support Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load ML Models (TF-IDF)
queue_model = None
prio_model = None
try:
    queue_model = joblib.load(os.path.join(BASE_DIR, '../models/tfidf_queue_model.joblib'))
    prio_model = joblib.load(os.path.join(BASE_DIR, '../models/tfidf_priority_model.joblib'))
    print("[OK] Triage models loaded successfully.")
except FileNotFoundError:
    print("WARNING: Triage models not found. /triage endpoint will be unavailable.")
except Exception as e:
    print(f"WARNING: Error loading triage models: {e}")

# 4. LLM & Database Setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            cited_ticket_ids TEXT,
            tool_used TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS eval_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            expected_queue_or_ids TEXT,
            passed INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 5. System Prompt — Grounding rules for the chatbot
SYSTEM_PROMPT = """You are a Support Intelligence Assistant that helps support agents find solutions from past support tickets.

STRICT RULES — you must follow these at all times:

1. TOOL USAGE: You have two tools available:
   - search_similar_tickets: Search the ticket archive for similar past problems and their solutions
   - query_stats: Query database statistics about ticket volumes, queues, and priorities

2. GROUNDING: Every answer MUST come from the results of your tools. You must NEVER invent, fabricate, or guess a solution. If a tool returns results, base your answer on those results and cite the ticket IDs.

3. CITATIONS: When providing information from tickets, ALWAYS cite the ticket ID(s) used. Format: "Based on ticket #1234..." or mention the IDs clearly.

4. NO SIMILAR CASE: If search_similar_tickets returns empty results or the results don't seem relevant to the question, you MUST say: "I don't have a similar case for that in our ticket archive." Do NOT make up an answer.

5. SCOPE: You ONLY answer questions related to support tickets and support operations. If someone asks a question unrelated to support (like general knowledge, math, trivia), politely decline and explain: "I can only help with support-related questions from our ticket archive. I'm not able to answer general knowledge questions."

6. FOLLOW-UPS: When the user asks a follow-up question that references earlier context (e.g., "what if that doesn't work?"), use the conversation history to understand what "that" refers to, then search for more specific solutions.

7. PRIVACY: Never reveal customer personal data (names, phone numbers, account numbers) even if they appear in ticket answers. Replace them with placeholders like [customer] or [redacted].

8. STATISTICS: When asked about ticket counts, volumes, or distributions, use the query_stats tool. Report the exact numbers returned.

9. FORMAT: Provide clear, concise answers. Summarize the solution from the ticket rather than dumping the entire raw answer. Always mention which ticket(s) you're drawing from."""

# 6. Pydantic Models
class TriageRequest(BaseModel):
    subject: str
    body: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]

# 7. Basic Endpoints (Phase 4)
@app.get("/health")
def health_check():
    status = {"status": "healthy", "database": "unknown", "vector_store": "unknown", "models": "unknown"}
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        status["database"] = "connected"
    except Exception as e:
        status["status"] = "degraded"
        status["database"] = f"error: {str(e)}"
    
    # Check FAISS index
    try:
        import rag.retriever as retriever_module
        retriever_module.load_rag_resources()
        if retriever_module.index is not None:
            status["vector_store"] = "connected"
        else:
            status["vector_store"] = "not loaded"
    except Exception as e:
        status["vector_store"] = f"error: {str(e)}"
    
    # Check models
    status["models"] = "loaded" if queue_model is not None else "not loaded"
    
    return status

@app.get("/stats")
def get_stats():
    conn = get_db_connection()
    
    # Queue distribution
    cursor = conn.execute("SELECT tag_1 AS queue, COUNT(*) as count FROM tickets GROUP BY tag_1 ORDER BY count DESC")
    queue_dist = {row["queue"]: row["count"] for row in cursor.fetchall()}
    
    # Priority distribution
    cursor = conn.execute("SELECT priority, COUNT(*) as count FROM tickets GROUP BY priority ORDER BY count DESC")
    priority_dist = {row["priority"]: row["count"] for row in cursor.fetchall()}
    
    # Total tickets
    cursor = conn.execute("SELECT COUNT(*) as total FROM tickets")
    total = cursor.fetchone()["total"]
    
    # Type distribution
    cursor = conn.execute("SELECT type, COUNT(*) as count FROM tickets GROUP BY type ORDER BY count DESC")
    type_dist = {row["type"]: row["count"] for row in cursor.fetchall()}
    
    # Top queues with priority breakdown
    cursor = conn.execute("""
        SELECT tag_1 AS queue, priority, COUNT(*) as count 
        FROM tickets 
        GROUP BY tag_1, priority 
        ORDER BY count DESC
    """)
    queue_priority = {}
    for row in cursor.fetchall():
        q = row["queue"]
        if q not in queue_priority:
            queue_priority[q] = {}
        queue_priority[q][row["priority"]] = row["count"]
    
    # Average answer length per top queue
    cursor = conn.execute("""
        SELECT tag_1 AS queue, AVG(LENGTH(answer)) as avg_len 
        FROM tickets 
        GROUP BY tag_1 
        ORDER BY avg_len DESC 
        LIMIT 10
    """)
    avg_answer_len = {row["queue"]: round(row["avg_len"], 0) for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_tickets": total,
        "queue_distribution": queue_dist,
        "priority_distribution": priority_dist,
        "type_distribution": type_dist,
        "queue_priority_breakdown": queue_priority,
        "avg_answer_length_by_queue": avg_answer_len
    }

@app.get("/tickets")
def get_tickets(queue: Optional[str] = None, priority: Optional[str] = None, 
                type: Optional[str] = None, limit: int = 50, offset: int = 0):
    conn = get_db_connection()
    query = "SELECT id, subject, body, answer, priority, type, tag_1 AS queue FROM tickets WHERE 1=1"
    count_query = "SELECT COUNT(*) as total FROM tickets WHERE 1=1"
    params = []
    count_params = []
    
    if queue:
        query += " AND tag_1 = ?"
        count_query += " AND tag_1 = ?"
        params.append(queue)
        count_params.append(queue)
    if priority:
        query += " AND priority = ?"
        count_query += " AND priority = ?"
        params.append(priority)
        count_params.append(priority)
    if type:
        query += " AND type = ?"
        count_query += " AND type = ?"
        params.append(type)
        count_params.append(type)
    
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor = conn.execute(query, params)
    tickets = [dict(row) for row in cursor.fetchall()]
    
    cursor = conn.execute(count_query, count_params)
    total = cursor.fetchone()["total"]
    
    conn.close()
    return {"tickets": tickets, "total": total, "limit": limit, "offset": offset}

@app.post("/triage")
def triage_ticket(request: TriageRequest):
    if queue_model is None or prio_model is None:
        raise HTTPException(status_code=503, detail="Triage models not loaded. Please ensure Phase 3 models are trained.")
    
    text = f"{request.subject} {request.body}"
    
    queue_pred = queue_model.predict([text])[0]
    queue_probs = queue_model.predict_proba([text])[0]
    prio_pred = prio_model.predict([text])[0]
    prio_probs = prio_model.predict_proba([text])[0]
    
    # Get top 3 queue predictions
    queue_classes = queue_model.classes_
    top_queue_indices = queue_probs.argsort()[-3:][::-1]
    top_queues = [
        {"queue": queue_classes[i], "confidence": float(queue_probs[i])}
        for i in top_queue_indices
    ]
    
    return {
        "queue": queue_pred,
        "queue_confidence": float(max(queue_probs)),
        "priority": prio_pred,
        "priority_confidence": float(max(prio_probs)),
        "top_queues": top_queues
    }

# 8. Chatbot Endpoint (Phase 6)
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
        
    current_cited_ids = []
    current_tools_used = []

    def search_similar_tickets(query: str, queue_filter: str = None) -> str:
        """Search the support ticket archive for similar past problems and solutions. Use this whenever the customer or agent describes a technical issue or problem."""
        nonlocal current_cited_ids, current_tools_used
        current_tools_used.append("search_similar_tickets")
        
        # Normalize queue filter
        if queue_filter and queue_filter.lower() == "billing":
            queue_filter = "Billing"
        elif queue_filter and queue_filter.lower() == "payment":
            queue_filter = "Payment"
            
        results = retrieve(query=query, k=3, queue_filter=queue_filter)
        if not results:
            return "No similar tickets found in the archive for this query."
            
        snippets = []
        for r in results:
            current_cited_ids.append(str(r["ticket_id"]))
            snippets.append(
                f"Ticket #{r['ticket_id']} (Queue: {r['queue']}, Priority: {r['priority']})\n"
                f"Subject: {r['subject']}\n"
                f"Resolution: {r['answer']}"
            )
        return "\n---\n".join(snippets)

    def query_stats(queue: str = None, priority: str = None) -> str:
        """Query database statistics about ticket volumes, queues, and priorities. Use this when the user asks about numbers, counts, volumes, or distribution of tickets."""
        nonlocal current_tools_used
        current_tools_used.append("query_stats")
        conn = get_db_connection()
        
        query = "SELECT COUNT(*) as count FROM tickets WHERE 1=1"
        params = []
        if queue:
            if "billing" in queue.lower() and "payment" in queue.lower():
                query += " AND (tag_1 = 'Billing' OR tag_1 = 'Payment')"
            else:
                query += " AND tag_1 LIKE ?"
                params.append(f"%{queue}%")
        if priority:
            query += " AND LOWER(priority) = LOWER(?)"
            params.append(priority)
            
        cursor = conn.execute(query, params)
        count = cursor.fetchone()["count"]
        
        detail = f"Found {count} tickets matching filters (queue={queue}, priority={priority})."
        if queue and not priority:
            cursor = conn.execute(
                "SELECT priority, COUNT(*) as cnt FROM tickets WHERE tag_1 LIKE ? GROUP BY priority",
                [f"%{queue}%"]
            )
            breakdown = {row["priority"]: row["cnt"] for row in cursor.fetchall()}
            detail += f" Breakdown by priority: {breakdown}"
            
        conn.close()
        return detail

    # Build conversation history for multi-turn
    history = []
    for m in request.messages[:-1]:
        role = "user" if m.role == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part.from_text(text=m.content)]))

    try:
        chat = client.chats.create(
            model="gemini-3.1-flash-lite",
            history=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[search_similar_tickets, query_stats],
                temperature=0.0
            )
        )
        
        last_user_msg = request.messages[-1].content
        response = chat.send_message(last_user_msg)
        final_answer = response.text or "I wasn't able to find an answer."
    except Exception as e:
        print(f"Chat generation error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    unique_cited_ids = list(dict.fromkeys(current_cited_ids))
    tool_used_str = ",".join(current_tools_used) if current_tools_used else "None"

    # Log interaction to database
    try:
        conn = get_db_connection()
        user_msg = request.messages[-1].content
        conn.execute(
            "INSERT INTO chat_logs (session_id, role, message, tool_used) VALUES (?, ?, ?, ?)",
            (request.session_id, "user", user_msg, "None")
        )
        conn.execute(
            "INSERT INTO chat_logs (session_id, role, message, cited_ticket_ids, tool_used) VALUES (?, ?, ?, ?, ?)",
            (request.session_id, "assistant", final_answer, ",".join(unique_cited_ids), tool_used_str)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to log chat: {e}")

    return {
        "answer": final_answer,
        "cited_ticket_ids": unique_cited_ids,
        "tool_trace": tool_used_str
    }