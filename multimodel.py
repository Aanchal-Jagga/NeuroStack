"""
app.py

Full multimodal RAG pipeline:
- Download Flowers-102 dataset
- Save first N images locally
- Create / populate ChromaDB with OpenCLIP embeddings
- Query DB with text
- Use Ollama vision model (moondream:1.5b) to caption/desribe images
- Use Ollama text model (gemma:270m) to generate bouquet suggestions
"""

import os
import io
import time
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ChromaDB
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

# Ollama
import ollama

# ----------------- Config -----------------
DATASET_NAME = "huggan/flowers-102-categories"
DATA_DIR = Path("./dataset/flowers-102-categories")
DB_DIR = Path("./data/flower.db")          # path for chromadb persistent client
COLLECTION_NAME = "flowers_collection"
SAVE_COUNT = 500                           # number of images to save & index
VISION_MODEL = "moondream:1.5b"            # Ollama vision model (smallish)
TEXT_MODEL = "gemma:270m"                  # Ollama text model (Gemma 270m)
N_RESULTS = 2                              # images to retrieve per query

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.parent.mkdir(parents=True, exist_ok=True)

# ----------------- Helper: show image -----------------
def show_image(path_or_pil):
    if isinstance(path_or_pil, (str, Path)):
        img = Image.open(path_or_pil)
    else:
        img = path_or_pil
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# ----------------- Step 1: Download dataset & save images -----------------
def download_and_save_images(dataset_name=DATASET_NAME, save_dir=DATA_DIR, num_images=SAVE_COUNT):
    print(f"Loading dataset {dataset_name} from Hugging Face...")
    ds = load_dataset(dataset_name, split="train")  # using train split (contains many images)

    n = min(num_images, len(ds))
    print(f"Saving first {n} images to {save_dir} ...")

    for i in tqdm(range(n), desc="Saving images"):
        item = ds[i]
        img = item["image"]
        filename = save_dir / f"flower_{i+1:04d}.png"
        # some PIL images might be mode 'RGBA' - convert to RGB
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(filename)
        except Exception as e:
            # fallback: convert via bytes
            try:
                img_bytes = img.tobytes()
                Image.frombytes(img.mode, img.size, img_bytes).save(filename)
            except Exception:
                print(f"Failed saving image index {i}: {e}")
    print("Done saving images.")

# ----------------- Step 2: Setup ChromaDB and collection -----------------
def get_chroma_collection(path=DB_DIR, collection_name=COLLECTION_NAME):
    print("Starting ChromaDB PersistentClient...")
    # PersistentClient will create a local DB at the path
    client = chromadb.PersistentClient(path=str(path))
    # use image loader and OpenCLIP embedding fn
    embedding_fn = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    coll = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        data_loader=image_loader,
    )
    return client, coll

# ----------------- Step 3: Index images into collection -----------------
def index_images(collection, images_folder=DATA_DIR):
    # collect files
    print("Collecting PNG files to index...")
    files = sorted([str(p) for p in Path(images_folder).glob("*.png")])
    if not files:
        raise RuntimeError(f"No images found in {images_folder}. Run download_and_save_images first.")

    # check if collection already has same count
    try:
        existing = collection.count()
    except Exception:
        existing = 0

    print(f"Collection currently has {existing} items. About to add {len(files)} files (if not present).")

    # We'll add only if not already added
    # A simple approach: if existing < files_count -> add all (dedupe left as exercise)
    if existing >= len(files):
        print("Collection already contains equal or more items than images on disk; skipping add.")
        return

    ids = [str(i) for i in range(len(files))]
    uris = files

    print("Adding images to collection (this may take a while while embeddings are computed)...")
    collection.add(ids=ids, uris=uris)
    print("Indexing done. Collection count:", collection.count())

# ----------------- Step 4: Query function -----------------
def query_db(collection, query_text: str, n_results: int = N_RESULTS) -> Dict:
    print(f"Querying ChromaDB for: '{query_text}' (n_results={n_results})")
    results = collection.query(query_texts=[query_text], n_results=n_results, include=["uris", "distances", "ids"])
    return results

# ----------------- Step 5: Use Ollama vision model to describe images -----------------
def describe_images_with_ollama(image_paths: List[str], vision_model: str = VISION_MODEL) -> List[str]:
    """
    Call Ollama vision model to get captions / descriptions for each image.
    Ollama's `chat` accepts images parameter (list of local file paths).
    """
    descriptions = []
    for p in image_paths:
        print(f"Describing image: {p}")
        try:
            # The Ollama python client chat call
            # messages: system / user roles. Use a user message requesting a brief descriptive caption.
            resp = ollama.chat(
                model=vision_model,
                messages=[{"role": "user", "content": "Provide a short descriptive caption of this flower image. Mention colors, petal shape, and notable features."}],
                images=[p],
            )
            # Response structure: resp['message']['content'] (string)
            desc = resp.get("message", {}).get("content", "").strip()
            descriptions.append(desc)
            print(" ->", desc)
        except Exception as e:
            print("Ollama vision call failed for", p, "->", e)
            descriptions.append("No description available.")
    return descriptions

# ----------------- Step 6: Use Gemma (text) to combine captions + query -----------------
def generate_suggestion_with_gemma(user_query: str, image_captions: List[str], text_model: str = TEXT_MODEL) -> str:
    """
    Build a prompt containing the user's query and the image captions,
    and ask gemma:270m to produce bouquet arrangement suggestions.
    """
    # create a succinct prompt
    prompt_lines = [
        "You are a talented florist. Given the user's request and the descriptions of two flower images, suggest a bouquet arrangement idea.",
        f"User request: {user_query}",
        "Image descriptions:"
    ]
    for i, c in enumerate(image_captions, start=1):
        prompt_lines.append(f"{i}. {c}")

    prompt_lines.append(
        "Provide: 1) a short bouquet idea (2-4 sentences), 2) suggested complementary flowers/colors, 3) a short reasoning sentence referencing the image descriptions."
    )
    prompt = "\n".join(prompt_lines)

    try:
        resp = ollama.chat(model=text_model, messages=[{"role": "user", "content": prompt}])
        text = resp.get("message", {}).get("content", "").strip()
        return text
    except Exception as e:
        print("Gemma call failed:", e)
        return "Error: Could not get suggestion from gemma."

# ----------------- Utility: print results and show images -----------------
def print_and_show_results(results):
    uris = results["uris"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    print("\nRetrieved images:")
    for idx, uri in enumerate(uris):
        print(f"ID: {ids[idx]}, Distance: {distances[idx]:.4f}, Path: {uri}")
        show_image(uri)

# ----------------- Main Flow -----------------
def main():
    # 1) Download dataset and save images (only if folder empty)
    pngs = list(DATA_DIR.glob("*.png"))
    if not pngs:
        print("Dataset images not found locally. Downloading dataset & saving images...")
        download_and_save_images()
    else:
        print(f"Found {len(pngs)} images in {DATA_DIR}")

    # 2) Setup chroma collection
    client, collection = get_chroma_collection()

    # 3) Index images (if not already)
    index_images(collection, DATA_DIR)

    # 4) Interactive loop: user queries
    print("\nReady. Enter queries (type 'exit' to quit).")
    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        # 5) Retrieve images
        results = query_db(collection, user_query, n_results=N_RESULTS)
        # show retrieved images (optional)
        print_and_show_results(results)

        # 6) Describe images with vision model
        image_paths = results["uris"][0]
        captions = describe_images_with_ollama(image_paths, vision_model=VISION_MODEL)

        # 7) Generate suggestion with gemma
        suggestion = generate_suggestion_with_gemma(user_query, captions, text_model=TEXT_MODEL)

        print("\n=== AI Florist Suggestion ===\n")
        print(suggestion)
        print("\n-----------------------------\n")

if __name__ == "__main__":
    main()
