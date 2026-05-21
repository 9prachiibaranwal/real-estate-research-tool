# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.
# Modified to use ChromaDB + SentenceTransformers + Groq API for RAG pipeline

import os
import re
import json
import requests
import urllib3
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, CrossEncoder

import chromadb
from groq import Groq

# Suppress noisy SSL warnings when SSL fallback is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Langfuse observability — imported AFTER load_dotenv() per Langfuse best practices
from langfuse import get_client as get_langfuse_client

# Constants
CHUNK_SIZE = 1000
COLLECTION_NAME = "real_estate"

# Global components (Reverted to original architecture for thread safety on Windows)
embedding_model = None
reranker_model = None
chroma_client = None
collection = None
groq_client = None


def initialize_components():
    """Initialize embedding model, ChromaDB, and Groq client."""
    global embedding_model, reranker_model, chroma_client, collection, groq_client

    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if reranker_model is None:
        print("Loading reranker model...")
        reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    if chroma_client is None:
        # Use EphemeralClient for cloud deployment (in-memory, no disk writes)
        chroma_client = chromadb.EphemeralClient()

    # Always refresh collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"[DEBUG] Collection '{COLLECTION_NAME}' has {collection.count()} documents (in-memory)")

    if groq_client is None:
        # Try Streamlit secrets first (for cloud), then fallback to .env
        try:
            api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found. Set it in Streamlit secrets or .env file")
        groq_client = Groq(api_key=api_key)


def validate_url(url):
    """Validate that a string is a properly formatted HTTP/HTTPS URL."""
    url_pattern = re.compile(
        r'^https?://'
        r'[a-zA-Z0-9.-]+'
        r'(?:\.[a-zA-Z]{2,})'
        r'(?:/[^\s]*)?$'
    )
    if not url_pattern.match(url):
        return False, f"Invalid URL format: '{url}'. URLs must start with http:// or https://"
    return True, None


def load_url_content(url):
    """Scrape content from a URL using BeautifulSoup.
    
    Handles SSL certificate errors (common on Windows) by retrying
    with verification disabled as a fallback.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt_verify in (True, False):
        try:
            response = requests.get(
                url, headers=headers, timeout=15, verify=attempt_verify
            )
            response.raise_for_status()

            if not attempt_verify:
                print(f"[WARNING] SSL verification disabled for {url}")

            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator="\n")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n\n".join(lines)

        except requests.exceptions.SSLError:
            if attempt_verify:
                print(f"[WARNING] SSL verification failed for {url}, retrying without verification...")
                continue
            else:
                print(f"Error loading {url}: SSL verification failed even without verify")
                return None
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
    """Scrape data from URLs and store it in the vector db."""
    valid_urls = []
    for url in urls:
        is_valid, error = validate_url(url)
        if is_valid:
            valid_urls.append(url)
        else:
            yield f"⚠️ Skipping invalid URL: {url}"

    if not valid_urls:
        yield "❌ No valid URLs provided."
        return

    yield "Initializing Components...✅"
    initialize_components()

    yield "Resetting vector store...✅"
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

    for url in valid_urls:
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
    
    # Create embeddings using the global model safely
    embeddings = embedding_model.encode(all_chunks).tolist()

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end]
        )

    yield f"✅ Done! Added {len(all_chunks)} chunks to vector database."


def generate_answer(query, n_retrieve=10, top_k=4, strict_mode=False):
    if collection is None:
        raise RuntimeError("Collection is None -- initialize_components() was not called")
    if collection.count() == 0:
        raise RuntimeError("Vector database is empty. Please process URLs first.")

    try:
        langfuse = get_langfuse_client()
    except Exception:
        langfuse = None

    query_embedding = embedding_model.encode([query]).tolist()

    retrieval_ctx = None
    if langfuse:
        retrieval_ctx = langfuse.start_as_current_observation(
            as_type="span", name="chromadb-retrieval",
            input={"query": query, "n_results": n_retrieve},
        )

    try:
        if retrieval_ctx:
            retrieval_obs = retrieval_ctx.__enter__()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_retrieve,
            include=["documents", "metadatas", "distances"]
        )
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if retrieval_ctx:
            retrieval_obs.update(output={"chunks_retrieved": len(documents)})
    finally:
        if retrieval_ctx:
            retrieval_ctx.__exit__(None, None, None)

    rerank_ctx = None
    if langfuse:
        rerank_ctx = langfuse.start_as_current_observation(
            as_type="span", name="cross-encoder-reranking",
            input={"query": query, "num_candidates": len(documents)},
        )

    try:
        if rerank_ctx:
            rerank_obs = rerank_ctx.__enter__()

        query_doc_pairs = [[query, doc] for doc in documents]
        rerank_scores = reranker_model.predict(query_doc_pairs)

        ranked = sorted(
            zip(documents, metadatas, distances, rerank_scores),
            key=lambda x: x[3],
            reverse=True
        )

        MAX_COSINE_DISTANCE = 0.5
        context_parts = []
        sources_set = set()
        for doc, meta, dist, score in ranked[:top_k]:
            if dist <= MAX_COSINE_DISTANCE:
                source = meta.get("source", "Unknown")
                sources_set.add(source)
                context_parts.append(f"[Source: {source}]\n{doc}")

        if not context_parts:
            no_info_answer = "I don't have enough information in the provided articles to answer this question."
            if rerank_ctx:
                rerank_obs.update(output={"top_k_selected": 0, "early_exit": True})
            return no_info_answer, "", ""

        context = "\n\n".join(context_parts)

        if rerank_ctx:
            rerank_obs.update(output={"top_k_selected": len(context_parts)})
    finally:
        if rerank_ctx:
            rerank_ctx.__exit__(None, None, None)

    if strict_mode:
        prompt = f"""You are answering a question using ONLY the context below.
Follow these rules strictly:
1. ONLY state facts that are EXPLICITLY written in the context.
2. Do NOT infer, assume, or add any information beyond what is provided.
3. If the context does not contain the answer, respond ONLY with: "I don't have enough information in the provided articles to answer this question."
4. Quote specific details exactly as they appear.

Context:
{context}

Question: {query}

Answer:"""
    else:
        prompt = f"""Based on the following context from news articles, answer the question accurately.
Include specific details like dates, numbers, and rates mentioned in the context.
At the end, mention which sources you used.
If the context does not contain enough information, say "I don't have enough information in the provided articles to answer this question".

Context:
{context}

Question: {query}

Answer:"""

    gen_ctx = None
    if langfuse:
        gen_ctx = langfuse.start_as_current_observation(
            as_type="generation", name="groq-answer-generation",
            model="llama-3.3-70b-versatile",
            input=[{"role": "system", "content": "Research assistant"}, {"role": "user", "content": prompt}],
            metadata={"n_retrieve": n_retrieve, "top_k": top_k, "strict_mode": strict_mode},
        )

    try:
        if gen_ctx:
            gen_obs = gen_ctx.__enter__()

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that answers questions based ONLY on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 if strict_mode else 0.2,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        sources_str = "\n".join(sources_set)

        if gen_ctx:
            gen_obs.update(
                output=answer,
                usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
            )
    finally:
        if gen_ctx:
            gen_ctx.__exit__(None, None, None)

    return answer, sources_str, context


def check_groundedness(answer, context):
    try:
        langfuse = get_langfuse_client()
    except Exception:
        langfuse = None

    prompt = f"""You are a groundedness evaluator. Your job is to check whether an ANSWER
is fully supported by the given CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "verdict": "grounded" or "hallucinated",
  "unsupported_claims": ["list of specific claims not found in context"],
  "confidence": 0.0 to 1.0
}}"""

    gen_ctx = None
    if langfuse:
        gen_ctx = langfuse.start_as_current_observation(
            as_type="generation", name="groundedness-check",
            model="llama-3.3-70b-versatile",
            input=[{"role": "system", "content": "Strict factual evaluator"}, {"role": "user", "content": prompt}],
        )

    try:
        if gen_ctx:
            gen_obs = gen_ctx.__enter__()

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a strict factual evaluator. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=300
        )

        try:
            result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            result = {"verdict": "grounded", "unsupported_claims": [], "confidence": 0.5}

        if gen_ctx:
            gen_obs.update(
                output=result,
                usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens},
            )
    finally:
        if gen_ctx:
            gen_ctx.__exit__(None, None, None)

    return result


def generate_answer_with_healing(query, max_retries=1):
    initialize_components()

    try:
        langfuse = get_langfuse_client()
    except Exception:
        langfuse = None

    trace_ctx = None
    if langfuse:
        trace_ctx = langfuse.start_as_current_observation(
            as_type="span", name="self-healing-rag",
            input={"query": query},
            metadata={"max_retries": max_retries},
        )

    try:
        if trace_ctx:
            trace_obs = trace_ctx.__enter__()

        for attempt in range(max_retries + 1):
            is_retry = attempt > 0
            n_retrieve = 20 if is_retry else 10
            top_k = 8 if is_retry else 4

            answer, sources, context = generate_answer(query, n_retrieve, top_k, is_retry)

            if context == "":
                evaluation = {"verdict": "grounded", "unsupported_claims": [], "confidence": 1.0, "attempt": attempt + 1, "no_context": True}
                if trace_ctx:
                    trace_obs.update(output={"answer": answer, "verdict": "grounded", "attempt": attempt + 1, "no_context": True})
                return answer, sources, evaluation

            evaluation = check_groundedness(answer, context)
            evaluation["attempt"] = attempt + 1

            if langfuse:
                try:
                    langfuse.score_current_trace(
                        name="groundedness",
                        value=1.0 if evaluation["verdict"] == "grounded" else 0.0,
                        comment=f"Confidence: {evaluation.get('confidence', 'N/A')}, Attempt: {attempt + 1}",
                    )
                except Exception:
                    pass

            if evaluation["verdict"] == "grounded":
                if trace_ctx:
                    trace_obs.update(output={"answer": answer, "verdict": "grounded", "attempt": attempt + 1})
                return answer, sources, evaluation

            if attempt < max_retries:
                print(f"[WARNING] Hallucination detected (attempt {attempt + 1}), retrying with stricter settings...")
                continue

        fallback_result = (
            "I don't have enough information in the provided articles to answer this question accurately.",
            sources,
            {"verdict": "hallucinated", "unsupported_claims": evaluation.get("unsupported_claims", []), "confidence": 0.0, "attempt": max_retries + 1, "fallback": True}
        )
        if trace_ctx:
            trace_obs.update(output={"verdict": "hallucinated", "fallback": True, "attempt": max_retries + 1})
        return fallback_result
    finally:
        if trace_ctx:
            trace_ctx.__exit__(None, None, None)