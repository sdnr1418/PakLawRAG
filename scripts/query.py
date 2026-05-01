import os
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

_SCRIPTS = Path(__file__).parent
_STORE_PATH = str(_SCRIPTS / "../vectorstore_sections")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
GITHUB_MODELS_BASE_URL = os.getenv(
    "GITHUB_MODELS_BASE_URL",
    "https://models.github.ai/inference",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")

_vectorstore = None

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant specializing in Pakistani law.
Answer the user's question using ONLY the PPC sections provided below.
Be concise and cite the section numbers in your answer.
If the sections don't contain enough information to answer, say so.

Relevant PPC Sections:
{context}"""),
    ("human", "{question}"),
])


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            _STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def generate_answer(context: str, question: str) -> str:
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Missing GitHub Models token. Set GITHUB_TOKEN or GITHUB_MODELS_TOKEN."
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

    response = requests.post(
        f"{GITHUB_MODELS_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def query_vectorstore(query, k=3):
    docs = get_vectorstore().similarity_search(query, k=k)

    context = "\n\n".join(
        f"[Section {d.metadata['section_id']}]\n{d.metadata.get('original_text') or d.page_content}"
        for d in docs
    )

    print(f"\nQuery: {query}\n")
    print("=" * 80)
    answer = generate_answer(context, query)
    print(answer)
    print("=" * 80)
    print("\nSources:", ", ".join(f"§{d.metadata['section_id']}" for d in docs))


if __name__ == "__main__":
    query = input("Enter your legal question: ")
    query_vectorstore(query)
