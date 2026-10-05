import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

CV_FOLDER = "cvs"
DB_DIR = "chroma_db"
COLLECTION = "cvs"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_vector_store():
    # Start fresh each time so re-running doesn't create duplicate chunks
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    all_chunks = []
    for filename in os.listdir(CV_FOLDER):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(CV_FOLDER, filename), "r", encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.create_documents(
            texts=[text],
            metadatas=[{"source": filename}],
        )
        all_chunks.extend(chunks)

    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION,
    )
    print(f"Stored {len(all_chunks)} chunks from {len(os.listdir(CV_FOLDER))} files.")
    return db


def get_vector_store():

    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )


if __name__ == "__main__":
    build_vector_store()

    # Quick test: does retrieval find relevant CVs?
    db = get_vector_store()
    query = "Python backend engineer with AWS and Docker experience"
    results = db.similarity_search(query, k=3)

    print(f"\n--- TOP 3 CHUNKS for: '{query}' ---")
    for r in results:
        print(f"[{r.metadata['source']}] {r.page_content[:120].strip()}...")