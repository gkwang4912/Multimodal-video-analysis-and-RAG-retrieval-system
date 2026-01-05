import os
import sys
import sqlite3
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import mimetypes

# --- Configuration ---
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
MODEL_NAME = "openai/clip-vit-base-patch32"
CHUNK_SIZE = 800
OVERLAP = 100
DB_PATH = "rag_mm.db"
INDEX_TEXT_PATH = "text.index"
INDEX_IMAGE_PATH = "image.index"

# --- Globals for Models (Lazy loading) ---
clip_model = None
clip_processor = None

def load_models():
    global clip_model, clip_processor
    if clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(drop=False):
    conn = get_db_connection()
    if drop:
        conn.execute('DROP TABLE IF EXISTS metadata')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            faiss_id INTEGER,
            type TEXT,          -- 'text' or 'image'
            path TEXT,
            content TEXT,       -- text content or image path
            chunk_id TEXT       -- unique identifier for chunks
        )
    ''')
    conn.commit()
    conn.close()

def get_text_chunks(text, filename):
    chunks = []
    if not text:
        return chunks
    
    start = 0
    text_len = len(text)
    chunk_idx = 0
    
    while start < text_len:
        end = start + CHUNK_SIZE
        chunk_text = text[start:end]
        chunks.append({
            "id": f"{filename}#{chunk_idx}",
            "text": chunk_text
        })
        start += (CHUNK_SIZE - OVERLAP)
        chunk_idx += 1
    return chunks

def get_embeddings(texts=None, images=None):
    load_models()
    inputs = None
    if texts:
        inputs = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    elif images:
        inputs = clip_processor(images=images, return_tensors="pt")
    
    if inputs:
        with torch.no_grad():
            if texts:
                outputs = clip_model.get_text_features(**inputs)
            else:
                outputs = clip_model.get_image_features(**inputs)
            # Normalize embeddings
            embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            return embeddings.numpy()
    return None

def ingest(input_dir):
    # Initialize/Reset DB
    init_db(drop=True)
    
    text_embeddings = []
    image_embeddings = []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    files = []
    for root, _, filenames in os.walk(input_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
            
    print(f"Found {len(files)} files.")
    
    mimetypes.init()
    
    texts_to_embed = []
    images_to_embed = []
    
    text_metadata = []
    image_metadata = []
    
    for file_path in files:
        mime_type, _ = mimetypes.guess_type(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        is_image = mime_type and mime_type.startswith('image')
        is_text = mime_type and mime_type.startswith('text')
        
        if ext in ['.py', '.js', '.ts', '.md', '.json', '.yaml', '.yml', '.csv', '.tsv']:
            is_text = True
            
        try:
            if is_image:
                image = Image.open(file_path)
                images_to_embed.append(image)
                image_metadata.append(file_path)
                
            elif is_text:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    chunks = get_text_chunks(content, os.path.basename(file_path))
                    for chunk in chunks:
                        texts_to_embed.append(chunk['text'])
                        text_metadata.append({
                            "path": file_path,
                            "content": chunk['text'],
                            "chunk_id": chunk['id']
                        })
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"Embedding {len(texts_to_embed)} text chunks...")
    if texts_to_embed:
        emb_batches = []
        batch_size = 32
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i+batch_size]
            emb = get_embeddings(texts=batch)
            emb_batches.append(emb)
        if emb_batches:
            text_embeddings = np.vstack(emb_batches)

    print(f"Embedding {len(images_to_embed)} images...")
    if images_to_embed:
        emb_batches = []
        batch_size = 32
        for i in range(0, len(images_to_embed), batch_size):
            batch = images_to_embed[i:i+batch_size]
            emb = get_embeddings(images=batch)
            emb_batches.append(emb)
        if emb_batches:
            image_embeddings = np.vstack(emb_batches)

    if len(text_embeddings) > 0:
        d = text_embeddings.shape[1]
        text_index = faiss.IndexFlatIP(d)
        text_index.add(text_embeddings)
        faiss.write_index(text_index, INDEX_TEXT_PATH)
        
        for i, meta in enumerate(text_metadata):
            cursor.execute("INSERT INTO metadata (faiss_id, type, path, content, chunk_id) VALUES (?, ?, ?, ?, ?)",
                           (i, 'text', meta['path'], meta['content'], meta['chunk_id']))

    if len(image_embeddings) > 0:
        d = image_embeddings.shape[1]
        image_index = faiss.IndexFlatIP(d)
        image_index.add(image_embeddings)
        faiss.write_index(image_index, INDEX_IMAGE_PATH)
        
        for i, path in enumerate(image_metadata):
            cursor.execute("INSERT INTO metadata (faiss_id, type, path, content, chunk_id) VALUES (?, ?, ?, ?, ?)",
                           (i, 'image', path, path, f"img_{i}"))

    conn.commit()
    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_dir = "./input"
        print(f"No input directory specified. Defaulting to '{input_dir}'")
    else:
        input_dir = sys.argv[1]
        
    if not os.path.exists(input_dir):
        print(f"Error: Directory '{input_dir}' not found.")
        sys.exit(1)
        
    ingest(input_dir)
