# @Author: Dhaval Patel Copyrights Codebasics Inc. and LearnerX Pvt Ltd.
# Modified to use ChromaDB + SentenceTransformers + Groq API for RAG pipeline

import streamlit as st
from rag import process_urls, generate_answer_with_healing

st.title("AI Assistant Research Tool")

# Initialize session state to track URL processing across reruns
if "urls_processed" not in st.session_state:
    st.session_state.urls_processed = False

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

# Separate containers for status messages and query input
status_container = st.container()
query_container = st.container()

process_url_button = st.sidebar.button("Process URLs")
if process_url_button:
    urls = list(dict.fromkeys(url for url in (url1, url2, url3) if url != ''))
    if len(urls) == 0:
        status_container.warning("You must provide at least one valid URL")
    else:
        # Process URLs and track success via final status message
        processing_succeeded = False
        with status_container:
            for status in process_urls(urls):
                st.text(status)
                # Check the final message to determine success
                # process_urls yields "✅ ..." on success or "❌ ..." on failure
                if status.startswith("✅"):
                    processing_succeeded = True
                elif status.startswith("❌"):
                    processing_succeeded = False

        if processing_succeeded:
            st.session_state.urls_processed = True
            st.success("URLs have been processed, you can ask your query")
        else:
            st.session_state.urls_processed = False
            st.error("URL processing failed. Please check the URLs and try again.")

# Show persistent status if URLs were processed in a previous run
if st.session_state.urls_processed and not process_url_button:
    st.success("URLs have been processed, you can ask your query")

with query_container:
    query = st.text_input("Question")

query_button = st.button("Query")
if query_button and query:
    if not st.session_state.urls_processed:
        st.warning("You must process URLs first")
    else:
        try:
            with st.spinner("Generating and verifying answer..."):
                answer, sources, eval_info = generate_answer_with_healing(query)

            # Flush Langfuse traces to ensure they are sent
            try:
                from langfuse import get_client
                get_client().flush()
            except Exception:
                pass

            # Display groundedness status
            verdict = eval_info.get("verdict", "unknown")
            confidence = eval_info.get("confidence", "N/A")
            attempt = eval_info.get("attempt", 1)

            if eval_info.get("no_context"):
                st.info("ℹ️ No sufficiently relevant content found in the processed articles.")
            elif verdict == "grounded":
                st.success(f"✅ Answer is grounded (confidence: {confidence}, attempt: {attempt})")
            elif eval_info.get("fallback"):
                st.error(f"❌ Hallucination detected after {attempt} attempts — showing safe fallback")
            else:
                st.warning(f"⚠️ Answer may contain unsupported claims (attempt: {attempt})")

            st.header("Answer:")
            st.write(answer)

            if sources:
                st.subheader("Sources:")
                for source in sources.split("\n"):
                    st.write(source)

            # Show unsupported claims if any
            unsupported = eval_info.get("unsupported_claims", [])
            if unsupported:
                with st.expander("⚠️ Unsupported claims detected"):
                    for claim in unsupported:
                        st.write(f"- {claim}")

        except RuntimeError as e:
            st.error(f"Error: {str(e)}")
            st.session_state.urls_processed = False
elif query_button and not query:
    st.warning("Please enter your query")