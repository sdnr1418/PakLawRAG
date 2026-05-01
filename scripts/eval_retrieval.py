"""
Retrieval evaluation for the Pakistan Penal Code (PakLawRAG).

Metrics: Hit@k and MRR (Mean Reciprocal Rank).

NOTE: results reflect whatever is currently indexed in vectorstore_sections/.
Run build_vectorstore_sections.py first to ensure the index is up to date.

Run from any directory:
    python eval_retrieval.py
"""

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

_SCRIPTS = Path(__file__).parent
_STORE_PATH = str(_SCRIPTS / "../vectorstore_sections")

# ── vectorstore singleton ─────────────────────────────────────────────────────

_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("Loading vectorstore...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            _STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print("Vectorstore ready.\n")
    return _vectorstore


def retrieve(question: str, k: int = 5) -> list:
    return get_vectorstore().similarity_search(question, k=k)


# ── ground truth ──────────────────────────────────────────────────────────────
# Format: (query, [expected_section_ids], note)

TEST_CASES = [
    # Terminology gap — English common-law terms vs Islamized Urdu/Arabic titles
    ("What is the punishment for qatl-e-amd?",     ["300", "302"],        "terminology gap: murder → qatl-e-amd"),
    ("Define culpable homicide",                   ["318", "319"],        "terminology gap: culpable homicide → qatl-i-khata"),
    ("What is accidental killing?",                ["318", "321"],        "terminology gap: accidental killing → qatl-i-khata / qatl-bis-sabab"),
    ("Blood money compensation for killing",       ["323", "330"],        "terminology gap: blood money → diyat"),
    ("Hurt and grievous hurt",                     ["332", "337L"],       "terminology gap: hurt → shajjah / jurh series"),

    # Standard English queries — should work without normalization
    ("What constitutes theft?",                    ["378", "379"],        ""),
    ("Punishment for robbery",                     ["390", "392"],        ""),
    ("Definition of criminal breach of trust",     ["405", "406"],        ""),
    ("What is the punishment for rape?",           ["375", "376"],        ""),
    ("Kidnapping or abduction",                    ["359", "360", "362"], ""),
    ("Cheating and dishonestly inducing delivery", ["415", "420"],        ""),
    ("What is forgery?",                           ["463", "465"],        ""),
    ("Defamation",                                 ["499", "500"],        ""),
    ("Public servant taking bribe",                ["161"],               ""),
    ("Sedition against the state",                 ["124A"],              ""),
    ("What is the punishment for dacoity?",        ["391", "395"],        ""),
    ("Trespass criminal trespass",                 ["441", "447"],        ""),
]

K_VALUES = [1, 2, 3, 5]

# ── metrics ───────────────────────────────────────────────────────────────────

def hits_at_k(retrieved_ids: list, expected_ids: list, k: int) -> bool:
    return any(sid in expected_ids for sid in retrieved_ids[:k])


def reciprocal_rank(retrieved_ids: list, expected_ids: list) -> float:
    for rank, sid in enumerate(retrieved_ids, 1):
        if sid in expected_ids:
            return 1.0 / rank
    return 0.0


# ── runner ────────────────────────────────────────────────────────────────────

def run_eval(k_max: int = 5) -> None:
    results = []

    for question, expected, note in TEST_CASES:
        docs = retrieve(question, k=k_max)
        retrieved_ids = [doc.metadata.get("section_id", "") for doc in docs]
        rr   = reciprocal_rank(retrieved_ids, expected)
        hits = {k: hits_at_k(retrieved_ids, expected, k) for k in K_VALUES}
        results.append((question, expected, retrieved_ids, rr, hits, note))

    # ── per-query table ───────────────────────────────────────────────────────
    print(f"\n{'':2}{'Query':<48} {'Expected':<15} {'Retrieved@5':<32} {'RR':>4}  Note")
    print("-" * 115)
    for question, expected, retrieved, rr, hits, note in results:
        mark    = "✓" if any(hits.values()) else "✗"
        exp_str = ",".join(expected)
        ret_str = ",".join(retrieved[:5])
        print(f"{mark}  {question[:46]:<46} {exp_str:<15} {ret_str:<32} {rr:>4.2f}  {note}")

    # ── overall summary ───────────────────────────────────────────────────────
    print("\n--- Summary (all queries) ---")
    for k in K_VALUES:
        rate = sum(r[4][k] for r in results) / len(results)
        print(f"  Hit@{k}:  {rate:.1%}")
    print(f"  MRR:    {sum(r[3] for r in results) / len(results):.3f}")

    # ── terminology-gap split ─────────────────────────────────────────────────
    gap      = [r for r in results if r[5].startswith("terminology gap")]
    standard = [r for r in results if not r[5].startswith("terminology gap")]

    print(f"\n--- Terminology-gap queries ({len(gap)}) ---")
    for k in K_VALUES:
        rate = sum(r[4][k] for r in gap) / len(gap)
        print(f"  Hit@{k}:  {rate:.1%}")
    print(f"  MRR:    {sum(r[3] for r in gap) / len(gap):.3f}")

    print(f"\n--- Standard queries ({len(standard)}) ---")
    for k in K_VALUES:
        rate = sum(r[4][k] for r in standard) / len(standard)
        print(f"  Hit@{k}:  {rate:.1%}")
    print(f"  MRR:    {sum(r[3] for r in standard) / len(standard):.3f}")


if __name__ == "__main__":
    run_eval()
