import os
import pickle
import hashlib
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

UPLOAD_DIR = "uploads"
INDEX_DIR = "faiss_index"
HASH_FILE = os.path.join(INDEX_DIR, "pdf_hashes.pkl")
INDEX_FILE = os.path.join(INDEX_DIR, "index.pkl")

def hash_file(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append((i + 1, text))
    return texts

def chunk_text(text, metadata):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

def load_existing_hashes():
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_hashes(hashes):
    with open(HASH_FILE, "wb") as f:
        pickle.dump(hashes, f)

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    existing_hashes = load_existing_hashes()
    new_hashes = {}
    all_chunks = []

    for root, _, files in os.walk(UPLOAD_DIR):
        for file in files:
            if file.endswith(".pdf"):
                path = os.path.join(root, file)
                file_hash = hash_file(path)
                new_hashes[file] = file_hash

                print(f"🔍 Processing: {file}")
                pages = extract_text_from_pdf(path)
                for page_num, page_text in pages:
                    metadata = {
                        "source": path,
                        "filename": file,
                        "page": page_num
                    }
                    chunks = chunk_text(page_text, metadata)
                    all_chunks.extend(chunks)

    if all_chunks:
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if os.path.exists(INDEX_FILE):
            with open(INDEX_FILE, "rb") as f:
                vectorstore, tender_data = pickle.load(f)
            vectorstore.add_documents(all_chunks)
        else:
            vectorstore = FAISS.from_documents(all_chunks, embedder)
            tender_data = {}

        with open(INDEX_FILE, "wb") as f:
            pickle.dump((vectorstore, tender_data), f)

        save_hashes(new_hashes)
        print("✅ Index updated with new documents.")
    else:
        print("🟢 No new documents to process.")

if __name__ == "__main__":
    main()
