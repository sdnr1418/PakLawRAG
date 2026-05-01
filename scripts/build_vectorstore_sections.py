import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

_SCRIPTS = Path(__file__).parent
_OUTPUT_PATH = _SCRIPTS / "../output/ppc_sections.json"
_STORE_PATH = str(_SCRIPTS / "../vectorstore_sections")


def load_sections(json_path=None):
    if json_path is None:
        json_path = _OUTPUT_PATH
    with open(json_path, "r", encoding="utf-8") as f:
        sections = json.load(f)
    return sections


def convert_to_documents(sections):
    documents = []

    for sec in sections:
        section_id = sec["section_id"]
        original_text = sec["text"]
        # embed normalized_text so English equivalents of Urdu terms are indexed
        embed_text = sec.get("normalized_text") or original_text

        doc = Document(
            page_content=embed_text,
            metadata={
                "section_id":    section_id,
                "source":        f"PPC Section {section_id}",
                "original_text": original_text,
            }
        )

        documents.append(doc)

    return documents


def build_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    Path(_STORE_PATH).mkdir(exist_ok=True)
    vectorstore.save_local(_STORE_PATH)

    return vectorstore


if __name__ == "__main__":
    sections = load_sections()
    documents = convert_to_documents(sections)

    print(f"Loaded {len(sections)} parsed sections")
    print(f"Converted {len(documents)} documents")

    vectorstore = build_vectorstore(documents)

    print(f"Vector store created successfully at {_STORE_PATH}")