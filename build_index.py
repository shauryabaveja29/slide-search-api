import json
from sentence_transformers import SentenceTransformer
import faiss
import pickle

print("Starting...")

# Load the model
print("Loading AI model...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Load your slides
print("Loading slides...")
with open('slides.json', 'r') as f:
    slides = json.load(f)

print(f"Found {len(slides)} slides")

# Get descriptions
descriptions = [slide['what'] for slide in slides]

# Create embeddings
print("Creating embeddings (this takes 30 seconds)...")
embeddings = model.encode(descriptions)

# Save to index
print("Saving index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, 'slides_index.faiss')

with open('slides_metadata.pkl', 'wb') as f:
    pickle.dump(slides, f)

print("Done! Files created:")
print("- slides_index.faiss")

print("- slides_metadata.pkl")
