from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import pickle

app = Flask(__name__)

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
    
    # Search
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, 3)
    
    # Get results
    results = []
    for i, idx in enumerate(indices[0]):
        slide = slides[idx]
        results.append({
            "rank": i + 1,
            "slide_id": slide['slide_id'],
            "title": slide['title'],
            "what": slide['what']
        })
    
    return jsonify({
        "query": query,
        "results": results
    })

if __name__ == '__main__':

    app.run(port=5000)
