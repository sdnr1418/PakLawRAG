import os
import html
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

ROOT = Path(__file__).parent
STORE_PATH = str(ROOT / "vectorstore_sections")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
GITHUB_MODELS_BASE_URL = os.getenv(
    "GITHUB_MODELS_BASE_URL",
    "https://models.github.ai/inference",
)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant specializing in Pakistani law.

INSTRUCTIONS:
1. Answer the user's question using ONLY the PPC sections provided below.
2. Be concise, legally careful, and cite the PPC section numbers (e.g., Section 302, 376).
3. If multiple sections are relevant, combine them to provide a complete answer.
4. If the provided sections do NOT directly answer the question, say so clearly and explain what information IS available in these sections.
5. Do NOT make up or assume information from memory—only use what's in the sections below.

Relevant PPC Sections:
{context}"""),
    ("human", "{question}"),
])


st.set_page_config(
    page_title="PakLawRAG",
    page_icon="PL",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --bg: #07110f;
        --panel: rgba(10, 22, 20, 0.82);
        --panel-strong: rgba(12, 31, 28, 0.94);
        --line: rgba(142, 245, 214, 0.18);
        --line-strong: rgba(142, 245, 214, 0.34);
        --text: #e7f8f2;
        --muted: #9ab7ae;
        --accent: #54e3b2;
        --accent-2: #7bc8ff;
        --warn: #f7c66b;
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(84, 227, 178, 0.12), transparent 34%),
            linear-gradient(225deg, rgba(123, 200, 255, 0.10), transparent 36%),
            var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .app-hero {
        border-bottom: 1px solid var(--line);
        padding-bottom: 1.2rem;
        margin-bottom: 1.4rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: var(--accent);
        font-size: 0.86rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
    }

    .brand-dot {
        width: 0.62rem;
        height: 0.62rem;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 18px rgba(84, 227, 178, 0.65);
    }

    h1 {
        color: var(--text);
        font-size: 3rem !important;
        letter-spacing: 0 !important;
        margin: 0.45rem 0 0.4rem !important;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1rem;
        max-width: 760px;
        line-height: 1.65;
    }

    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 1.2rem 0 1.1rem;
    }

    .metric-box {
        border: 1px solid var(--line);
        background: rgba(9, 20, 18, 0.62);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.76rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1rem;
        font-weight: 700;
    }

    .stTextArea textarea {
        background: var(--panel-strong);
        color: var(--text);
        border: 1px solid var(--line-strong);
        border-radius: 8px;
        min-height: 150px;
        font-size: 1rem;
        line-height: 1.55;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px rgba(84, 227, 178, 0.38);
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid rgba(84, 227, 178, 0.62);
        background: linear-gradient(135deg, rgba(84, 227, 178, 0.96), rgba(123, 200, 255, 0.86));
        color: #04110e;
        font-weight: 800;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        border-color: rgba(231, 248, 242, 0.72);
        color: #04110e;
    }

    .answer-panel {
        border: 1px solid var(--line);
        background: rgba(8, 18, 16, 0.74);
        border-radius: 8px;
        padding: 1.1rem 1.2rem;
        min-height: 190px;
    }

    .panel-title {
        color: var(--accent);
        font-size: 0.84rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0;
        margin-bottom: 0.65rem;
    }

    .source-card {
        border: 1px solid var(--line);
        background: rgba(8, 18, 16, 0.70);
        border-radius: 8px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.8rem;
    }

    .source-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: var(--text);
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .source-rank {
        color: var(--accent-2);
        font-size: 0.84rem;
        white-space: nowrap;
    }

    .source-text {
        color: #cfe6de;
        line-height: 1.58;
        font-size: 0.92rem;
    }

    .empty-state {
        border: 1px dashed var(--line-strong);
        border-radius: 8px;
        padding: 2rem;
        color: var(--muted);
        text-align: center;
    }

    .notice {
        color: var(--warn);
        font-size: 0.86rem;
        line-height: 1.5;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 8px !important;
        background: rgba(8, 18, 16, 0.70) !important;
    }

    @media (max-width: 760px) {
        h1 {
            font-size: 2.1rem !important;
        }
        .metric-strip {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


@st.cache_resource(show_spinner=False)
def load_github_token():
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_MODELS_TOKEN")
    except Exception:
        return None


def retrieve_evidence(question: str, k: int):
    vectorstore = load_vectorstore()
    docs_and_scores = vectorstore.similarity_search_with_score(question, k=k)
    
    # Explicit section retrieval for common legal queries
    keywords_to_sections = {
        "murder": "302",
        "qatl": "302",
        "rape": "376",
        "dacoity": "396",
        "theft": "378",
        "assault": "354",
        "robbery": "390",
    }
    
    # Check if question contains keywords and add that section
    question_lower = question.lower()
    for keyword, section_id in keywords_to_sections.items():
        if keyword in question_lower:
            # Try to find and add this section if not already present
            existing_ids = {doc.metadata.get("section_id") for doc, _ in docs_and_scores}
            if section_id not in existing_ids:
                try:
                    # Search for the specific section
                    specific = vectorstore.similarity_search(f"Section {section_id}", k=1)
                    if specific:
                        docs_and_scores.insert(0, (specific[0], 0.99))
                except:
                    pass
            break
    
    return docs_and_scores


def generate_answer(question: str, docs_and_scores):
    docs = [doc for doc, _score in docs_and_scores]
    context = "\n\n".join(
        f"[Section {doc.metadata['section_id']}]\n"
        f"{doc.metadata.get('original_text') or doc.page_content}"
        for doc in docs
    )
    messages = PROMPT.format_messages(context=context, question=question)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": message.type, "content": message.content}
            for message in messages
        ],
        "temperature": 0,
        "max_tokens": 420,
    }
    for message in payload["messages"]:
        if message["role"] == "human":
            message["role"] = "user"

    token = load_github_token()
    if not token:
        raise RuntimeError(
            "Missing GitHub Models token. Add GITHUB_TOKEN in Streamlit Secrets."
        )

    response = requests.post(
        f"{GITHUB_MODELS_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def render_sources(docs_and_scores):
    for index, (doc, score) in enumerate(docs_and_scores, start=1):
        section_id = doc.metadata.get("section_id", "Unknown")
        source = doc.metadata.get("source", f"PPC Section {section_id}")
        text = doc.metadata.get("original_text") or doc.page_content
        preview = text if len(text) <= 2800 else text[:2800].rstrip() + "..."
        safe_preview = html.escape(preview).replace("\n", "<br>")
        safe_source = html.escape(source)

        with st.expander(f"Rank {index} - Section {section_id}", expanded=index == 1):
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-head">
                        <span>{safe_source}</span>
                        <span class="source-rank">FAISS score {score:.4f}</span>
                    </div>
                    <div class="source-text">{safe_preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = None
if "sources" not in st.session_state:
    st.session_state.sources = []


st.markdown(
    """
    <section class="app-hero">
        <div class="brand-row"><span class="brand-dot"></span><span>PakLawRAG</span></div>
        <h1>Pakistan Penal Code RAG</h1>
        <div class="hero-copy">
            Ask a legal question and inspect the exact PPC sections retrieved before the answer was generated.
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-strip">
        <div class="metric-box">
            <div class="metric-label">Corpus</div>
            <div class="metric-value">PPC section index</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Embedding</div>
            <div class="metric-value">{EMBED_MODEL}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Generator</div>
            <div class="metric-value">{LLM_MODEL}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown('<div class="panel-title">Query</div>', unsafe_allow_html=True)
    question = st.text_area(
        "Legal question",
        value=st.session_state.query,
        placeholder="Example: What is the punishment for qatl-e-amd?",
        label_visibility="collapsed",
    )

    controls = st.columns([0.48, 0.52], gap="medium")
    with controls[0]:
        top_k = st.slider("Retrieved sections", min_value=1, max_value=5, value=3)
    with controls[1]:
        st.write("")
        st.write("")
        ask = st.button("Analyze", use_container_width=True)

    st.markdown(
        """
        <div class="notice">
            The answer is generated only from retrieved PPC sections. Always verify citations against the evidence panel.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="panel-title">Answer</div>', unsafe_allow_html=True)
    if ask:
        clean_question = question.strip()
        if not clean_question:
            st.warning("Enter a question first.")
        else:
            st.session_state.query = clean_question
            with st.spinner("Retrieving sections and generating answer..."):
                try:
                    sources = retrieve_evidence(clean_question, top_k)
                    st.session_state.sources = sources
                    st.session_state.answer = generate_answer(clean_question, sources)
                except Exception as exc:
                    st.session_state.answer = None
                    st.error(
                        "Evidence retrieval may still work, but answer generation failed. "
                        "Check that `GITHUB_TOKEN` is configured in Streamlit Secrets, "
                        "has access to GitHub Models, and that the selected model "
                        f"`{LLM_MODEL}` is available."
                    )
                    st.caption(str(exc))

    if st.session_state.answer:
        safe_answer = html.escape(st.session_state.answer).replace("\n", "<br>")
        st.markdown(
            f"""
            <div class="answer-panel">
                {safe_answer}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="empty-state">
                Submit a question to generate a cited PPC answer.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

st.markdown('<div class="panel-title">Retrieved Evidence</div>', unsafe_allow_html=True)
if st.session_state.sources:
    render_sources(st.session_state.sources)
else:
    st.markdown(
        """
        <div class="empty-state">
            Retrieved PPC sections will appear here after a query.
        </div>
        """,
        unsafe_allow_html=True,
    )
