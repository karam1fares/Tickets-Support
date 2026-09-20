import sqlite3
import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '../data/warehouse.db')
INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')
META_PATH = os.path.join(BASE_DIR, 'metadata.pkl')

print("Loading data from warehouse...")
conn = sqlite3.connect(DB_PATH)
# Fetching id, queue, priority for metadata, and text for embedding
df = pd.read_sql_query("SELECT id, tag_1 AS queue, priority, subject, body, answer FROM tickets", conn)
conn.close()

# Drop rows missing crucial text
df = df.dropna(subset=['subject', 'body', 'answer'])

print("Formatting chunks...")
# Combine context (subject/body) with the answer to create rich chunks
df['chunk'] = "Subject: " + df['subject'] + "\nBody: " + df['body'] + "\nAnswer: " + df['answer']

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encoding texts (this may take a minute)...")
embeddings = model.encode(df['chunk'].tolist(), show_progress_bar=True, convert_to_numpy=True)

print("Building FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save metadata (mapping FAISS row IDs to ticket data)
metadata = df[['id', 'queue', 'priority', 'chunk', 'answer']].to_dict('records')

print("Saving index and metadata...")
faiss.write_index(index, INDEX_PATH)
with open(META_PATH, 'wb') as f:
    pickle.dump(metadata, f)

print(f"Successfully indexed {len(metadata)} tickets!")