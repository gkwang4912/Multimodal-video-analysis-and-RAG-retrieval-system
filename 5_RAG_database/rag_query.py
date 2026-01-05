import os
import sys
import sqlite3
import numpy as np
import faiss
import torch
import base64
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI

# --- Configuration ---
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
API_KEY = "lm-studio"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LLM_MODEL_NAME = "qwen/qwen3-vl-4b"
TOP_K_TEXT = 5
TOP_K_IMAGE = 3
DB_PATH = "rag_mm.db"
INDEX_TEXT_PATH = "text.index"
INDEX_IMAGE_PATH = "image.index"

# --- Globals for Models (Lazy loading) ---
clip_model = None
clip_processor = None
client = OpenAI(base_url=LM_STUDIO_URL, api_key=API_KEY)

def load_models():
    global clip_model, clip_processor
    if clip_model is None:
        print("Loading CLIP model...")
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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

def image_to_base64_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        mime_type = "image/jpeg"
        if image_path.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_path.lower().endswith(".webp"):
            mime_type = "image/webp"
        return f"data:{mime_type};base64,{base64_image}"

def query_rag_api(query_text):
    if not os.path.exists(INDEX_TEXT_PATH) or not os.path.exists(DB_PATH):
        return {"error": "Index not found"}

    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Embed Query
    query_vec = get_embeddings(texts=[query_text])
    
    # 2. Search Text Index
    top_text_results = []
    if os.path.exists(INDEX_TEXT_PATH):
        try:
            text_index = faiss.read_index(INDEX_TEXT_PATH)
            D, I = text_index.search(query_vec, TOP_K_TEXT)
            for idx, score in zip(I[0], D[0]):
                if idx != -1:
                    row = cursor.execute("SELECT * FROM metadata WHERE type='text' AND faiss_id=?", (int(idx),)).fetchone()
                    if row:
                        top_text_results.append({
                            "score": float(score),
                            "chunk_id": row['chunk_id'],
                            "path": row['path'],
                            "content": row['content']
                        })
        except Exception as e:
            print(f"Error searching text index: {e}")

    # 3. Search Image Index
    top_image_results = []
    if os.path.exists(INDEX_IMAGE_PATH):
        try:
            image_index = faiss.read_index(INDEX_IMAGE_PATH)
            D, I = image_index.search(query_vec, TOP_K_IMAGE)
            for idx, score in zip(I[0], D[0]):
                if idx != -1:
                    row = cursor.execute("SELECT * FROM metadata WHERE type='image' AND faiss_id=?", (int(idx),)).fetchone()
                    if row:
                        top_image_results.append({
                            "score": float(score),
                            "path": row['path']
                        })
        except Exception as e:
            print(f"Error searching image index: {e}")

    # Prep Context
    context_texts = []
    for item in top_text_results:
        context_texts.append(f"Source: {item['chunk_id']}\nContent: {item['content']}")

    image_urls = []
    images_base64 = []
    for item in top_image_results:
        try:
            url = image_to_base64_url(item['path'])
            image_urls.append(url)
            images_base64.append({
                "path": item['path'],
                "url": url
            })
        except Exception as e:
            print(f"Error loading image {item['path']}: {e}")

    # 4. Prompt Construction
    system_prompt = "你是一個影片分析專家。請根據提供的上下文（文字和圖片）回答用戶問題。如果不確定，請說不知道。回答時請引用來源（chunk_id 或 image filename）。"
    
    user_content = [{"type": "text", "text": f"Question: {query_text}\n\nContext:\n" + "\n---\n".join(context_texts)}]
    
    for url in image_urls:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url}
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    print("\n=== Calling LM Studio... ===")
    answer = ""
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME, 
            messages=messages,
            temperature=0.7,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        answer = f"Error calling LLM: {e}"

    conn.close()
    
    return {
        "answer": answer,
        "sources": {
            "text": top_text_results,
            "images": images_base64
        }
    }

def query_rag(query_text):
    result = query_rag_api(query_text)
    
    if "error" in result:
        print(result["error"])
        return

    print("\n=== Top Text Hits ===")
    for item in result["sources"]["text"]:
        print(f"[{item['score']:.4f}] {item['chunk_id']} ({item['path']})")

    print("\n=== Top Image Hits ===")
    for item in result["sources"]["images"]:
        print(f"Image: {item['path']}")

    print("\n=== Answer ===")
    print(result["answer"])

def main():
    # If explicit argument is provided, do one-shot query
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
        query_rag(query_text)
    else:
        # Interactive mode
        print("進入互動模式。請輸入問題，或按 Ctrl+C 離開。")
        while True:
            try:
                user_input = input("\n請輸入問題: ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    break
                if user_input.strip():
                    query_rag(user_input.strip())
            except KeyboardInterrupt:
                print("\nBye!")
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()
