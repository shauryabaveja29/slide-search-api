from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import pickle

app = Flask(__name__)
CORS(app)

print("Loading everything...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
index = faiss.read_index('slides_index.faiss')

with open('slides_metadata.pkl', 'rb') as f:
    slides = pickle.load(f)

print(f"Ready! Loaded {len(slides)} slides")

@app.route('/')
def home():
    return jsonify({
        "status": "API is working!",
        "total_slides": len(slides)
    })

@app.route('/search')
def search():
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400
    
    # Convert query to embedding
    query_embedding = model.encode([query]).astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(query_embedding, 3)
    
    # FAISS returns L2 distances (lower = more similar)
    # Convert to similarity score (higher = more similar)
    # Threshold: Only return if distance < 1.0 (you can adjust this)
    RELEVANCE_THRESHOLD = 1.5
    
    results = []
    for i, idx in enumerate(indices[0]):
        distance = distances[0][i]
        
        # Only include if relevant enough
        if distance < RELEVANCE_THRESHOLD:
            slide = slides[idx]
            results.append({
                "rank": i + 1,
                "slide_id": slide['slide_id'],
                "title": slide['title'],
                "what": slide['what'],
                "relevance_score": float(1 / (1 + distance))  # Convert to 0-1 score
            })
    
    # If no relevant results
    if len(results) == 0:
        return jsonify({
            "query": query,
            "results": [],
            "message": "No relevant slides found. Try different keywords."
        })
    
    return jsonify({
        "query": query,
        "results": results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


