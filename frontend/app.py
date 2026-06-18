"""
Streamlit frontend for the AWS Customer Agreement RAG Q&A system.
Runs as a separate process and calls the FastAPI backend over HTTP.
"""

import requests
import streamlit as st
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AWS Agreement Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AWS Customer Agreement — Q&A")

tab_chat, tab_analytics, tab_ingest = st.tabs(["💬 Chat", "📊 Analytics", "⚙️ Setup"])


# ── Setup tab ──────────────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Document Ingestion")
    st.info(
        "The PDF must already be placed at `data/aws_customer_agreement.pdf` "
        "relative to where you started the backend. Click the button below to "
        "parse it and build the vector index."
    )
    pdf_path = st.text_input("PDF path (relative to backend working directory)", value="data/aws_customer_agreement.pdf")
    if st.button("Ingest Document"):
        with st.spinner("Parsing and embedding…"):
            try:
                resp = requests.post(f"{API_BASE}/ingest", json={"pdf_path": pdf_path}, timeout=120)
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"{data['message']} ({data['chunks_created']} chunks created)")
                else:
                    st.error(f"Error {resp.status_code}: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Make sure FastAPI is running on port 8000.")

    st.divider()
    if st.button("Check backend health"):
        try:
            h = requests.get(f"{API_BASE}/health", timeout=5).json()
            if h.get("index_loaded"):
                st.success("Backend is up and the vector index is loaded.")
            else:
                st.warning("Backend is up but no index loaded — run Ingest first.")
        except Exception:
            st.error("Backend unreachable.")


# ── Chat tab ───────────────────────────────────────────────────────────────────
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Source passages used"):
                    for src in msg["sources"]:
                        st.markdown(f"**Chunk {src['chunk_id']}**")
                        st.text(src["text"])
                        st.divider()

    query = st.chat_input("Ask a question about the AWS Customer Agreement…")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ask",
                        json={"query": query},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        latency = data["response_time_ms"]
                        found = data["answer_found"]

                        if not found:
                            st.warning(answer)
                        else:
                            st.markdown(answer)

                        st.caption(f"Response time: {latency:.0f} ms")

                        if sources:
                            with st.expander("Source passages used"):
                                for src in sources:
                                    st.markdown(f"**Chunk {src['chunk_id']}**")
                                    st.text(src["text"])
                                    st.divider()

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })
                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", "")
                        st.error(f"Backend error: {detail}")
                    else:
                        st.error(f"Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the backend. Is FastAPI running on port 8000?")


# ── Analytics tab ──────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("Usage Analytics")

    if st.button("Refresh analytics"):
        st.session_state.pop("analytics_data", None)

    if "analytics_data" not in st.session_state:
        try:
            resp = requests.get(f"{API_BASE}/analytics", timeout=10)
            if resp.status_code == 200:
                st.session_state.analytics_data = resp.json()
            else:
                st.error(f"Could not load analytics: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach the backend.")

    data = st.session_state.get("analytics_data")
    if data:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Queries", data["total_queries"])
        col2.metric("Avg Latency (ms)", f"{data['avg_response_latency_ms']:.1f}")
        unanswered_count = sum(q["count"] for q in data["unanswered_queries"])
        col3.metric("Unanswered Queries", unanswered_count)

        st.divider()

        if data["most_frequent_queries"]:
            st.subheader("Most Frequently Asked Questions")
            df_freq = pd.DataFrame(data["most_frequent_queries"])
            df_freq["query_short"] = df_freq["query"].str[:60] + "…"
            fig = px.bar(
                df_freq,
                x="count",
                y="query_short",
                orientation="h",
                labels={"count": "Times Asked", "query_short": "Query"},
                title="Top 10 Queries by Frequency",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No queries logged yet.")

        if data["unanswered_queries"]:
            st.subheader("Out-of-Scope / Unanswered Queries")
            df_ua = pd.DataFrame(data["unanswered_queries"])
            st.dataframe(df_ua, use_container_width=True)
        else:
            st.success("All logged queries were answered from the document.")
