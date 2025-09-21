import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import re

print("Starting vector store creation...")

try:
    # 1. Load the detailed project descriptions
    with open('projects.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Chunk the text by project using "---" as a separator
    chunks = text.split('---')
    project_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    if not project_chunks:
        raise ValueError("No project chunks found. Check projects.txt for '---' separators.")
    print(f"Found and chunked {len(project_chunks)} projects.")

    # 3. Load the embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)... This may take a moment.")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Embedding model loaded.")

    # 4. Create embeddings for each project chunk
    print("Creating embeddings for project chunks...")
    embeddings = model.encode(project_chunks, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32') # FAISS requires float32
    print(f"Embeddings created with shape: {embeddings.shape}")

    # 5. Build the FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"FAISS index built. Total vectors in index: {index.ntotal}")

    # 6. Save the index and the text chunks for the main app to use
    faiss.write_index(index, 'faiss_index.bin')
    with open('project_chunks.txt', 'w', encoding='utf-8') as f:
        for chunk in project_chunks:
            f.write(chunk + "\n===\n") # Use a unique separator for easy splitting later

    print("\n✅ Vector store created successfully!")
    print("   - faiss_index.bin")
    print("   - project_chunks.txt")

except FileNotFoundError:
    print("❌ ERROR: `projects.txt` not found. Please create it and add your project details.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")