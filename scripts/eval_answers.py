"""
Answer-level evaluation for PakLawRAG.

This script checks whether the final LLM answer is factually adequate and
properly cited. It does not compare answer text against a gold answer by
semantic similarity, because legally correct answers can be phrased many ways.

Run from any directory:
    python scripts/eval_answers.py

Optional environment variables:
    ANSWER_MODEL=your-model-name
    MAX_ANSWER_TOKENS=180
    GITHUB_TOKEN=your_github_models_token
    GITHUB_MODELS_BASE_URL=https://models.github.ai/inference
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

_SCRIPTS = Path(__file__).parent
_ROOT = _SCRIPTS.parent
_STORE_PATH = str(_ROOT / "vectorstore_sections")
_REPORT_PATH = _ROOT / "output" / "answer_eval_report.json"
_MAX_SECTION_CHARS = 1800

load_dotenv()

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gemma3:4b")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "180"))
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "5") or "5")
GITHUB_MODELS_BASE_URL = os.getenv(
    "GITHUB_MODELS_BASE_URL",
    "https://models.github.ai/inference",
)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_MODELS_TOKEN")

_vectorstore = None


def looks_like_github_token(value: str) -> bool:
    return isinstance(value, str) and any(value.startswith(prefix) for prefix in ["ghp_", "github_pat_", "gho_", "ghu_"])


def use_github_model(model: str) -> bool:
    if looks_like_github_token(model):
        raise RuntimeError(
            "ANSWER_MODEL or JUDGE_MODEL appears to be a token, not a model name. "
            "Set GITHUB_TOKEN to your token and ANSWER_MODEL/JUDGE_MODEL to a GitHub model name like openai/gpt-4.1-mini."
        )
    return bool(GITHUB_TOKEN and "/" in model)


def github_chat_completion(model: str, messages, temperature: int, max_tokens: int) -> str:
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "Missing GitHub Models token. Set GITHUB_TOKEN or GITHUB_MODELS_TOKEN."
        )

    payload_messages = []
    for message in messages:
        if isinstance(message, dict):
            role = message["role"]
            content = message["content"]
        else:
            role = "user" if getattr(message, "type", None) == "human" else getattr(message, "type", None)
            content = getattr(message, "content", "")
        if role == "human":
            role = "user"
        payload_messages.append({"role": role, "content": content})

    payload = {
        "model": model,
        "messages": payload_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        f"{GITHUB_MODELS_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        body = response.text.strip()
        if response.status_code == 429:
            raise RuntimeError(
                "GitHub Models API rate limited (429). "
                "Check your token, retry after a short delay, or use a different model/token. "
                f"Response body: {body}"
            ) from exc
        raise RuntimeError(
            f"GitHub Models request failed ({response.status_code}). Response body: {body}"
        ) from exc
    return response.json()["choices"][0]["message"]["content"].strip()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a legal assistant specializing in Pakistani law.
Answer the user's question using ONLY the PPC sections provided below.
Be concise and cite the section numbers in your answer.
If the sections don't contain enough information to answer, say so.

Relevant PPC Sections:
{context}"""),
    ("human", "{question}"),
])


TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "defamation_basic",
        "question": "What is defamation under PPC?",
        "expected_sections": ["499", "500"],
        "required_facts": [
            [["imputation", "imputes"], ["person"]],
            [["harm"], ["reputation"]],
            [["section 500", "500"], ["punishment", "punished"]],
        ],
        "forbidden_claims": [
            r"\bonly\b.*\bcivil\b",
            r"\bsection\s+499\b.*\bpunishment\b",
        ],
    },
    {
        "id": "robbery_punishment",
        "question": "What is the punishment for robbery?",
        "expected_sections": ["392"],
        "required_facts": [
            [["robbery"]],
            [["rigorous imprisonment", "imprisonment"]],
            [["ten years", "10 years"]],
        ],
        "forbidden_claims": [
            r"\bdeath\b",
            r"\blife imprisonment\b",
        ],
    },
    {
        "id": "theft_definition",
        "question": "What constitutes theft under PPC?",
        "expected_sections": ["378", "379"],
        "required_facts": [
            [["dishonestly"]],
            [["movable property"]],
            [["possession"]],
            [["without consent", "without that person's consent", "without his consent"]],
        ],
        "forbidden_claims": [
            r"\bimmovable property\b",
        ],
    },
    {
        "id": "criminal_breach_of_trust",
        "question": "Define criminal breach of trust.",
        "expected_sections": ["405"],
        "required_facts": [
            [["entrusted", "entrustment"]],
            [["property"]],
            [["dishonestly"]],
            [["misappropriates", "misappropriation", "converts", "conversion", "uses", "disposes"]],
        ],
        "forbidden_claims": [
            r"\brequires\b.*\bforce\b",
        ],
    },
    {
        "id": "forgery_definition",
        "question": "What is forgery under PPC?",
        "expected_sections": ["463"],
        "required_facts": [
            [["false document", "false electronic record", "false electronic document"]],
            [["intent"]],
            [["damage", "injury", "fraud", "claim"]],
        ],
        "forbidden_claims": [
            r"\balways\b.*\bdeath\b",
        ],
    },
    {
        "id": "qatl_e_amd_punishment",
        "question": "What is the punishment for qatl-e-amd?",
        "expected_sections": ["302"],
        "required_facts": [
            [["qatl-e-amd", "qatl-i-amd", "intentional murder", "intentional killing"]],
            [["death"], ["qisas"]],
            [["ta'zir", "tazir", "imprisonment for life", "life imprisonment"]],
            [["twenty-five years", "25 years"]],
        ],
        "forbidden_claims": [
            r"\bonly\b.*\bfine\b",
        ],
    },
    {
        "id": "diyat_basic",
        "question": "What is diyat?",
        "expected_sections": ["323", "330"],
        "required_facts": [
            [["diyat"]],
            [["compensation", "blood money"]],
            [["heirs", "wali", "victim"]],
        ],
        "forbidden_claims": [
            r"\bimprisonment only\b",
        ],
    },
    {
        "id": "sedition_basic",
        "question": "What is sedition against the state?",
        "expected_sections": ["124A"],
        "required_facts": [
            [["hatred", "contempt", "disaffection"]],
            [["government", "state"]],
            [["words", "signs", "visible representation", "spoken", "written"]],
        ],
        "forbidden_claims": [
            r"\bprivate person\b.*\bonly\b",
        ],
    },
    {
        "id": "criminal_trespass",
        "question": "What is criminal trespass?",
        "expected_sections": ["441"],
        "required_facts": [
            [["enters", "entry", "remain"]],
            [["property"]],
            [["intent"]],
            [["offence", "intimidate", "insult", "annoy"]],
        ],
        "forbidden_claims": [
            r"\bpublic property only\b",
        ],
    },
    {
        "id": "known_weak_bribery",
        "question": "Public servant taking bribe",
        "expected_sections": ["161"],
        "required_facts": [
            [["public servant"]],
            [["gratification", "bribe", "reward"]],
            [["official act", "official function", "public duty"]],
        ],
        "forbidden_claims": [
            r"\bnot\b.*\boffence\b",
        ],
    },
    {
        "id": "rape_definition",
        "question": "What is rape under PPC?",
        "expected_sections": ["375"],
        "required_facts": [
            [["rape"]],
            [["penetrates", "penetration", "inserts", "manipulates", "applies"]],
            [["without consent", "against will", "under sixteen", "unable to communicate consent"]],
        ],
        "forbidden_claims": [
            r"\bonly\b.*\bwoman\b",
            r"\bconsent\b.*\balways\b.*\bdefence\b",
        ],
    },
    {
        "id": "rape_punishment",
        "question": "What is the punishment for rape under PPC?",
        "expected_sections": ["376"],
        "required_facts": [
            [["rape"]],
            [["death", "imprisonment"]],
            [["fine"]],
        ],
        "forbidden_claims": [
            r"\bonly\b.*\bfine\b",
        ],
    },
    {
        "id": "cheating_definition",
        "question": "What is cheating under PPC?",
        "expected_sections": ["415"],
        "required_facts": [
            [["deceives", "deception"]],
            [["fraudulently", "dishonestly"]],
            [["induces", "induce"]],
            [["property", "consent", "act", "omit"]],
        ],
        "forbidden_claims": [
            r"\brequires\b.*\bphysical force\b",
        ],
    },
    {
        "id": "cheating_delivery_property",
        "question": "What is the punishment for cheating and dishonestly inducing delivery of property?",
        "expected_sections": ["420"],
        "required_facts": [
            [["cheating"]],
            [["dishonestly"], ["delivery of property", "deliver property"]],
            [["imprisonment"], ["seven years", "7 years"]],
            [["fine"]],
        ],
        "forbidden_claims": [
            r"\bdeath\b",
            r"\blife imprisonment\b",
        ],
    },
    {
        "id": "dacoity_punishment",
        "question": "What is the punishment for dacoity?",
        "expected_sections": ["395"],
        "required_facts": [
            [["dacoity"]],
            [["imprisonment for life", "life imprisonment", "rigorous imprisonment"]],
            [["ten years", "10 years"]],
            [["fine"]],
        ],
        "forbidden_claims": [
            r"\bonly\b.*\bfine\b",
        ],
    },
    {
        "id": "kidnapping_from_pakistan",
        "question": "What is kidnapping from Pakistan?",
        "expected_sections": ["360"],
        "required_facts": [
            [["conveys", "takes"]],
            [["beyond the limits of pakistan", "beyond pakistan", "out of pakistan"]],
            [["without consent"]],
        ],
        "forbidden_claims": [
            r"\bwithin pakistan only\b",
        ],
    },
    {
        "id": "abduction_definition",
        "question": "What is abduction under PPC?",
        "expected_sections": ["362"],
        "required_facts": [
            [["force", "compels", "deceitful means"]],
            [["induces", "induce"]],
            [["go from any place", "go from a place"]],
        ],
        "forbidden_claims": [
            r"\bmust\b.*\bminor\b",
        ],
    },
    {
        "id": "wrongful_restraint",
        "question": "What is wrongful restraint?",
        "expected_sections": ["339"],
        "required_facts": [
            [["obstructs", "obstruction"]],
            [["voluntarily"]],
            [["prevent", "prevents"]],
            [["proceed", "right to proceed"]],
        ],
        "forbidden_claims": [
            r"\brequires\b.*\bconfinement\b",
        ],
    },
    {
        "id": "wrongful_confinement",
        "question": "What is wrongful confinement?",
        "expected_sections": ["340"],
        "required_facts": [
            [["wrongfully restrains", "wrongful restraint"]],
            [["prevent", "prevents"]],
            [["beyond certain circumscribing limits", "circumscribing limits", "limits"]],
        ],
        "forbidden_claims": [
            r"\bonly\b.*\bpublic servant\b",
        ],
    },
    {
        "id": "criminal_intimidation",
        "question": "What is criminal intimidation under PPC?",
        "expected_sections": ["503"],
        "required_facts": [
            [["threatens", "threat"]],
            [["injury"]],
            [["person", "reputation", "property"]],
            [["intent", "cause alarm", "omit", "do any act"]],
        ],
        "forbidden_claims": [
            r"\brequires\b.*\bactual injury\b",
        ],
    },
]


@dataclass
class EvalResult:
    id: str
    question: str
    expected_sections: list[str]
    retrieved_sections: list[str]
    answer: str
    retrieval_hit: bool
    citation_score: str
    fact_score: str
    forbidden_claims_found: list[str]
    passed: bool
    missing_citations: list[str]
    missing_facts: list[int]


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = FAISS.load_local(
            _STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def retrieve(question: str, k: int = 3):
    return get_vectorstore().similarity_search(question, k=k)


def answer_question(question: str, docs) -> str:
    context = "\n\n".join(
        f"[Section {d.metadata['section_id']}]\n"
        f"{(d.metadata.get('original_text') or d.page_content)[:_MAX_SECTION_CHARS]}"
        for d in docs
    )
    messages = PROMPT.format_messages(context=context, question=question)
    if use_github_model(ANSWER_MODEL):
        return github_chat_completion(ANSWER_MODEL, messages, 0, MAX_ANSWER_TOKENS)
    llm = ChatOllama(model=ANSWER_MODEL, temperature=0, num_predict=MAX_ANSWER_TOKENS)
    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content.strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def cites_section(answer: str, section_id: str) -> bool:
    escaped = re.escape(section_id)
    patterns = [
        rf"\bsection\s+{escaped}\b",
        rf"\bsec\.?\s+{escaped}\b",
        rf"§\s*{escaped}\b",
        rf"\b{escaped}\b",
    ]
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)


def fact_present(answer_norm: str, fact_groups: list[list[str]]) -> bool:
    """A fact passes when every group has at least one matching phrase."""
    for alternatives in fact_groups:
        if not any(phrase.lower() in answer_norm for phrase in alternatives):
            return False
    return True


def grade_case(case: dict[str, Any]) -> EvalResult:
    docs = retrieve(case["question"], k=3)
    retrieved_sections = [d.metadata.get("section_id", "") for d in docs]
    answer = answer_question(case["question"], docs)
    answer_norm = normalize(answer)

    expected_sections = case["expected_sections"]
    retrieval_hit = any(section in retrieved_sections for section in expected_sections)

    missing_citations = [
        section for section in expected_sections
        if not cites_section(answer, section)
    ]
    citation_hits = len(expected_sections) - len(missing_citations)

    missing_facts = []
    for index, fact in enumerate(case["required_facts"], start=1):
        if not fact_present(answer_norm, fact):
            missing_facts.append(index)
    fact_hits = len(case["required_facts"]) - len(missing_facts)

    forbidden_claims_found = [
        pattern for pattern in case.get("forbidden_claims", [])
        if re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
    ]

    passed = (
        retrieval_hit
        and not missing_citations
        and not missing_facts
        and not forbidden_claims_found
    )

    return EvalResult(
        id=case["id"],
        question=case["question"],
        expected_sections=expected_sections,
        retrieved_sections=retrieved_sections,
        answer=answer,
        retrieval_hit=retrieval_hit,
        citation_score=f"{citation_hits}/{len(expected_sections)}",
        fact_score=f"{fact_hits}/{len(case['required_facts'])}",
        forbidden_claims_found=forbidden_claims_found,
        passed=passed,
        missing_citations=missing_citations,
        missing_facts=missing_facts,
    )


def run_eval() -> list[EvalResult]:
    results = []
    for case in TEST_CASES[:EVAL_LIMIT]:
        print(f"\nRunning {case['id']}: {case['question']}", flush=True)
        result = grade_case(case)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  {status} | retrieved={','.join(result.retrieved_sections)} "
            f"| citations={result.citation_score} | facts={result.fact_score}"
        )
        if result.missing_citations:
            print(f"  missing citations: {', '.join(result.missing_citations)}")
        if result.missing_facts:
            print(f"  missing fact numbers: {result.missing_facts}")
        if result.forbidden_claims_found:
            print(f"  forbidden claims: {result.forbidden_claims_found}")

    return results


def save_report(results: list[EvalResult]) -> None:
    _REPORT_PATH.parent.mkdir(exist_ok=True)
    payload = {
        "summary": {
            "total": len(results),
            "passed": sum(r.passed for r in results),
            "failed": sum(not r.passed for r in results),
            "pass_rate": sum(r.passed for r in results) / len(results),
        },
        "results": [asdict(result) for result in results],
    }
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {_REPORT_PATH}")


if __name__ == "__main__":
    results = run_eval()
    save_report(results)
    passed = sum(r.passed for r in results)
    print(f"\nAnswer eval pass rate: {passed}/{len(results)} ({passed / len(results):.1%})")
