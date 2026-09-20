"""
Seed the eval_questions table with the acceptance test questions from the requirements.
Run once: python -m scripts.seed_eval_questions
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'warehouse.db')

EVAL_QUESTIONS = [
    {
        "question": "A customer's VPN keeps dropping after they reset their password — has this happened before?",
        "expected_queue_or_ids": "Technical,Network,Security"
    },
    {
        "question": "what if restarting the client doesn't fix it?",
        "expected_queue_or_ids": "follow-up — requires multi-turn context"
    },
    {
        "question": "How many high-priority tickets do we have in the Billing and Payments queue?",
        "expected_queue_or_ids": "query_stats tool — Billing + Payment queues"
    },
    {
        "question": "Find a similar case for a customer asking for a refund on a duplicate charge, only in the Billing queue.",
        "expected_queue_or_ids": "search_similar_tickets with queue_filter=Billing"
    },
    {
        "question": "My quantum entanglement photon beam resonance calibrator is malfunctioning on my Mars colony outpost — has this happened before?",
        "expected_queue_or_ids": "NO_SIMILAR_CASE — must decline"
    },
    {
        "question": "What's the capital of France?",
        "expected_queue_or_ids": "DECLINE — not support-related"
    },
]

def seed():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eval_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            expected_queue_or_ids TEXT,
            passed INTEGER DEFAULT 0
        )
    ''')
    
    # Check if already seeded
    cursor.execute("SELECT COUNT(*) FROM eval_questions")
    count = cursor.fetchone()[0]
    if count > 0:
        print(f"eval_questions already has {count} rows. Clearing and re-seeding...")
        cursor.execute("DELETE FROM eval_questions")
    
    for q in EVAL_QUESTIONS:
        cursor.execute(
            "INSERT INTO eval_questions (question, expected_queue_or_ids) VALUES (?, ?)",
            (q["question"], q["expected_queue_or_ids"])
        )
    
    conn.commit()
    print(f"[OK] Seeded {len(EVAL_QUESTIONS)} eval questions.")
    
    # Verify
    cursor.execute("SELECT id, question, expected_queue_or_ids FROM eval_questions")
    for row in cursor.fetchall():
        print(f"  [{row[0]}] {row[1][:60]}... -> {row[2]}")
    
    conn.close()

if __name__ == "__main__":
    seed()
