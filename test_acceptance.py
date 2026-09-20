import os
import sqlite3
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types
from rag.retriever import retrieve

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a Support Intelligence Assistant that helps support agents find solutions from past support tickets.

STRICT RULES — you must follow these at all times:

1. TOOL USAGE: You have two tools available:
   - search_similar_tickets: Search the ticket archive for similar past problems and their solutions
   - query_stats: Query database statistics about ticket volumes, queues, and priorities

2. GROUNDING: Every answer MUST come from the results of your tools. You must NEVER invent, fabricate, or guess a solution. If a tool returns results, base your answer on those results and cite the ticket IDs.

3. CITATIONS: When providing information from tickets, ALWAYS cite the ticket ID(s) used. Format: "Based on ticket #1234..." or "Ticket #1234: ...".

4. NO SIMILAR CASE: If search_similar_tickets returns empty results or no similar tickets found, you MUST say: "I don't have a similar case for that in our ticket archive." Do NOT make up an answer.

5. SCOPE: You ONLY answer questions related to customer support tickets and support operations. If someone asks a question unrelated to support (like general knowledge, trivia, capital cities, math), politely decline and state: "I can only help with support-related questions from our ticket archive. I'm not able to answer general knowledge questions."

6. FOLLOW-UPS: When the user asks a follow-up question that references earlier context (e.g., "what if restarting the client doesn't fix it?"), use the conversation history to understand the issue, then use the search tool to find further solutions.

7. PRIVACY: Never reveal customer personal data (names, phone numbers, email addresses, account numbers) even if they appear in ticket answers. Replace them with placeholders like [Customer Name] or [Account Number].

8. STATISTICS: When asked about ticket counts, volumes, or distributions, use the query_stats tool. Report the exact numbers returned.

9. FORMAT: Provide concise, helpful answers summarizing the verified resolution from the ticket."""

def get_db_connection():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/warehouse.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def run_chat(messages):
    current_cited_ids = []
    current_tools_used = []

    def search_similar_tickets(query: str, queue_filter: str = None) -> str:
        """Search the support ticket archive for similar past problems and solutions."""
        nonlocal current_cited_ids, current_tools_used
        current_tools_used.append("search_similar_tickets")
        
        # Handle queue aliases
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
                f"Resolution: {r['answer']}\n"
            )
        return "\n---\n".join(snippets)

    def query_stats(queue: str = None, priority: str = None) -> str:
        """Query database statistics about ticket volumes, queues, and priorities."""
        nonlocal current_tools_used
        current_tools_used.append("query_stats")
        conn = get_db_connection()
        
        # Build query
        query = "SELECT COUNT(*) as count FROM tickets WHERE 1=1"
        params = []
        if queue:
            # Handle combinations like "Billing and Payments"
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

    history = []
    for m in messages[:-1]:
        role = "user" if m["role"] == "user" else "model"
        history.append(types.Content(role=role, parts=[types.Part.from_text(text=m["content"])]))

    chat = client.chats.create(
        model="gemini-3.6-flash",
        history=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[search_similar_tickets, query_stats],
            temperature=0.0
        )
    )

    last_user_msg = messages[-1]["content"]
    response = chat.send_message(last_user_msg)
    
    return {
        "answer": response.text,
        "cited_ticket_ids": list(dict.fromkeys(current_cited_ids)), # remove duplicates preserving order
        "tool_trace": ",".join(current_tools_used) if current_tools_used else "None"
    }

if __name__ == "__main__":
    print("Testing 1: VPN question")
    r1 = run_chat([{"role": "user", "content": "A customer's VPN keeps dropping after they reset their password has this happened before?"}])
    print("Answer 1:", r1["answer"][:200], "...")
    print("Citations 1:", r1["cited_ticket_ids"])
    print("Tools 1:", r1["tool_trace"])
    print("-" * 50)

    print("Testing 2: Multi-turn follow up")
    r2 = run_chat([
        {"role": "user", "content": "A customer's VPN keeps dropping after they reset their password has this happened before?"},
        {"role": "assistant", "content": r1["answer"]},
        {"role": "user", "content": "what if restarting the client doesn't fix it?"}
    ])
    print("Answer 2:", r2["answer"][:200], "...")
    print("Citations 2:", r2["cited_ticket_ids"])
    print("Tools 2:", r2["tool_trace"])
    print("-" * 50)

    print("Testing 3: Stats question")
    r3 = run_chat([{"role": "user", "content": "How many high-priority tickets do we have in the Billing and Payments queue?"}])
    print("Answer 3:", r3["answer"])
    print("Tools 3:", r3["tool_trace"])
    print("-" * 50)

    print("Testing 4: Refund in Billing queue")
    r4 = run_chat([{"role": "user", "content": "Find a similar case for a customer asking for a refund on a duplicate charge, only in the Billing queue."}])
    print("Answer 4:", r4["answer"][:200], "...")
    print("Citations 4:", r4["cited_ticket_ids"])
    print("Tools 4:", r4["tool_trace"])
    print("-" * 50)

    print("Testing 5: Made-up scenario")
    r5 = run_chat([{"role": "user", "content": "My quantum entanglement photon beam resonance calibrator is malfunctioning on my Mars colony outpost - has this happened before?"}])
    print("Answer 5:", r5["answer"])
    print("Citations 5:", r5["cited_ticket_ids"])
    print("Tools 5:", r5["tool_trace"])
    print("-" * 50)

    print("Testing 6: Off-topic capital of France")
    r6 = run_chat([{"role": "user", "content": "What's the capital of France?"}])
    print("Answer 6:", r6["answer"])
    print("Tools 6:", r6["tool_trace"])
