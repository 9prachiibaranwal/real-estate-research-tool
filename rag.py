# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.
# Modified to use Haystack framework with Groq API

import os
import requests
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

from unstructured.partition.html import partition_html
from sentence_transformers import SentenceTransformer

import chromadb
from groq import Groq

load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = str(Path(__file__).parent / "resources" / "vectorstore")
COLLECTION_NAME = "real_estate"

# Global components
embedding_model = None
chroma_client = None
collection = None
groq_client = None


def initialize_components():
    """Initialize embedding model, ChromaDB, and Groq client."""
    global embedding_model, chroma_client, collection, groq_client

    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if chroma_client is None:
        # Ensure the vectorstore directory exists
        Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)

        chroma_client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    if groq_client is None:
        # Try Streamlit secrets first (for cloud), then fallback to .env
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found. Set it in Streamlit secrets or .env file")
        groq_client = Groq(api_key=api_key)


def load_url_content(url):
    """Scrape content from a URL using unstructured."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        elements = partition_html(text=response.text)
        text = "\n\n".join([str(el) for el in elements])
        return text
    except Exception as e:
        print(f"Error loading {url}: {e}")
        return None


def split_text(text, chunk_size=CHUNK_SIZE):
    """Split text into chunks using a simple recursive approach."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = [text]

    for sep in separators:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                parts = chunk.split(sep)
                current = ""
                for part in parts:
                    if len(current) + len(part) + len(sep) <= chunk_size:
                        current = current + sep + part if current else part
                    else:
                        if current:
                            new_chunks.append(current.strip())
                        current = part
                if current:
                    new_chunks.append(current.strip())
            else:
                new_chunks.append(chunk)
        chunks = new_chunks

    return [c for c in chunks if c.strip()]


def process_urls(urls):
    """
    This function scrapes data from URLs and stores it in a vector db.
    :param urls: input urls
    :return: yields status messages
    """
    yield "Initializing Components...✅"
    initialize_components()

    yield "Resetting vector store...✅"
    # Delete existing collection and recreate
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    global collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    yield "Loading data from URLs...✅"
    all_chunks = []
    all_metadatas = []
    all_ids = []

    for url in urls:
        text = load_url_content(url)
        if text:
            chunks = split_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadatas.append({"source": url})
                all_ids.append(str(uuid4()))

    if not all_chunks:
        yield "❌ No content could be loaded from the provided URLs."
        return

    yield f"Splitting text into {len(all_chunks)} chunks...✅"

    yield "Creating embeddings and adding to vector database...✅"
    # Create embeddings
    embeddings = embedding_model.encode(all_chunks).tolist()

    # Add to ChromaDB in batches (Chroma has a limit per batch)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end]
        )

    yield f"Done! Added {len(all_chunks)} chunks to vector database...✅"


def generate_answer(query):
    """Generate an answer using retrieval from ChromaDB + Groq LLM."""
    if collection is None or collection.count() == 0:
        raise RuntimeError("Vector database is not initialized or empty")

    # Create query embedding
    query_embedding = embedding_model.encode([query]).tolist()

    # Retrieve relevant documents from ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=4,
        include=["documents", "metadatas"]
    )

    # Build context from retrieved documents
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []
    sources_set = set()
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "Unknown")
        sources_set.add(source)
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""Based on the following context from news articles, answer the question accurately.
Include specific details like dates, numbers, and rates mentioned in the context.
At the end, mention which sources you used.

Context:
{context}

Question: {query}

Answer:"""

    # Call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful research assistant that answers questions based on provided context. Always cite your sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.9,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    sources_str = "\n".join(sources_set)

    return answer, sources_str


if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    for status in process_urls(urls):
        print(status)

    answer, sources = generate_answer(
        "Tell me what was the 30 year fixed mortgage rate along with the date?"
    )
    print(f"\nAnswer: {answer}")
    print(f"\nSources: {sources}")