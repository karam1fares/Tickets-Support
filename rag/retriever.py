import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, 'faiss_index.bin')
META_PATH = os.path.join(BASE_DIR, 'metadata.pkl')

# Distance threshold — L2 distances above this are considered irrelevant
# For all-MiniLM-L6-v2 with IndexFlatL2, typical relevant results have distance < 1.2
# Distances > 1.5 are usually unrelated
RELEVANCE_THRESHOLD = 1.5

# Initialize as None, load lazily to avoid overhead if imported without use
index = None
metadata = None
model = None

def load_rag_resources():
    global index, metadata, model
    if index is None:
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, 'rb') as f:
            metadata = pickle.load(f)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"[OK] RAG resources loaded: {index.ntotal} vectors in FAISS index")

def retrieve(query: str, k: int = 3, queue_filter: str = None):
    """
    Retrieve the top-k most similar tickets from the FAISS index.
    
    Args:
        query: The search query string.
        k: Number of results to return (default 3).
        queue_filter: Optional queue name to filter results.
    
    Returns:
        List of dicts with ticket_id, queue, priority, subject, body, answer, distance.
        Returns empty list if no relevant results found (distance threshold applied).
    """
    load_rag_resources()
    
    # Over-fetch to ensure we have enough after filtering
    fetch_k = k * 5 if queue_filter else k * 2
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, fetch_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:  # FAISS returns -1 if there aren't enough vectors
            continue
        
        distance = float(distances[0][i])
        
        # Apply distance threshold — skip irrelevant results
        if distance > RELEVANCE_THRESHOLD:
            continue
            
        row_meta = metadata[idx]
        
        # Apply optional queue filter
        if queue_filter and row_meta.get('queue', '') != queue_filter:
            continue
            
        results.append({
            "ticket_id": row_meta['id'],
            "queue": row_meta.get('queue', 'Unknown'),
            "priority": row_meta.get('priority', 'Unknown'),
            "subject": row_meta.get('chunk', '').split('\n')[0].replace('Subject: ', '') if 'chunk' in row_meta else '',
            "answer": row_meta.get('answer', ''),
            "distance": distance,
            "relevance_score": round(max(0, 1 - distance / RELEVANCE_THRESHOLD) * 100, 1)
        })
        
        if len(results) >= k:
            break
            
    return results

if __name__ == "__main__":
    # Quick sanity check
    print("Testing retrieval with queue filter:")
    results = retrieve("Customer wants a refund for a duplicate charge", k=2, queue_filter="Billing")
    for r in results:
        print(f"  Ticket #{r['ticket_id']} (queue={r['queue']}, dist={r['distance']:.3f}, relevance={r['relevance_score']}%)")
    
    print("\nTesting retrieval without filter:")
    results = retrieve("VPN keeps dropping after password reset", k=3)
    for r in results:
        print(f"  Ticket #{r['ticket_id']} (queue={r['queue']}, dist={r['distance']:.3f}, relevance={r['relevance_score']}%)")
    
    print("\nTesting with made-up query (should return empty):")
    results = retrieve("quantum entanglement photon beam resonance calibration on Mars", k=3)
    print(f"  Results: {len(results)} (should be 0 or very few)")