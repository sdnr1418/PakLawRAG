import json
import re
from pathlib import Path

_SCRIPTS = Path(__file__).parent

term_map = {
    # ── Intentional killing (murder) ──────────────────────────────────────────
    "qatl-i-amd": [
        "intentional murder", "murder", "wilful killing",
        "intentional killing", "homicide"
    ],
    "qatl i amd": [
        "intentional murder", "murder", "wilful killing",
        "intentional killing", "homicide"
    ],
    "qatl-e-amd": [
        "intentional murder", "murder", "wilful killing",
        "intentional killing", "homicide"
    ],

    # ── Semi-intentional killing ───────────────────────────────────────────────
    "qatl shibh-i-amd": [
        "semi-intentional killing", "quasi-intentional homicide",
        "unintentional death with harmful intent"
    ],
    "qatl shibh i amd": [
        "semi-intentional killing", "quasi-intentional homicide",
        "unintentional death with harmful intent"
    ],
    "qatl-e-shibh-i-amd": [
        "semi-intentional killing", "quasi-intentional homicide",
        "unintentional death with harmful intent"
    ],

    # ── Accidental / culpable homicide ────────────────────────────────────────
    "qatl-i-khata": [
        "culpable homicide", "accidental killing", "manslaughter",
        "unintentional killing", "negligent homicide", "homicide by mistake"
    ],
    "qatl i khata": [
        "culpable homicide", "accidental killing", "manslaughter",
        "unintentional killing", "negligent homicide", "homicide by mistake"
    ],
    "qatl-e-khata": [
        "culpable homicide", "accidental killing", "manslaughter",
        "unintentional killing", "negligent homicide", "homicide by mistake"
    ],

    # ── Killing by indirect cause ─────────────────────────────────────────────
    "qatl-bis-sabab": [
        "constructive homicide", "killing by indirect cause",
        "causing death indirectly"
    ],
    "qatl bis sabab": [
        "constructive homicide", "killing by indirect cause",
        "causing death indirectly"
    ],

    # ── Generic qatl (must come after all specific qatl-* entries) ────────────
    "qatl": ["killing", "homicide", "murder"],

    # ── Blood money / compensation ────────────────────────────────────────────
    "diyat": [
        "blood money", "financial compensation for killing",
        "compensation to legal heirs", "death compensation"
    ],
    "arsh": [
        "injury compensation", "blood money for hurt",
        "specified compensation", "financial remedy for bodily harm"
    ],
    "daman": [
        "compensation for hurt", "damages for injury",
        "financial remedy"
    ],

    # ── Retaliation ───────────────────────────────────────────────────────────
    "qisas": [
        "retaliation", "retribution", "eye for an eye",
        "equal punishment", "right of retaliation"
    ],

    # ── Punishment types ──────────────────────────────────────────────────────
    "ta'zir": ["discretionary punishment", "judge-determined punishment"],
    "tazir":  ["discretionary punishment", "judge-determined punishment"],
    "hadd":   ["fixed punishment", "mandatory punishment", "prescribed punishment"],

    # ── Pardon / settlement ───────────────────────────────────────────────────
    "afw":  ["pardon", "waiver of retaliation", "forgiveness by victim's family"],
    "sulh": ["compounding", "settlement", "compromise", "mutual agreement"],

    # ── Legal heir / guardian ─────────────────────────────────────────────────
    "wali": ["legal heir", "legal guardian", "victim's next of kin"],

    # ── Duress / coercion ─────────────────────────────────────────────────────
    "ikrah-i-tam":   ["complete coercion", "absolute duress"],
    "ikrah-i-naqis": ["incomplete coercion", "partial duress"],
    "ikrah":         ["coercion", "duress", "compulsion"],

    # ── Wound / hurt series ───────────────────────────────────────────────────
    "shajjah": [
        "head wound", "face wound", "injury to head or face",
        "grievous hurt to head"
    ],
    "jurh": [
        "body wound", "bodily injury", "hurt", "physical injury"
    ],
    "jaifah": [
        "penetrating wound", "deep body wound",
        "wound penetrating body cavity"
    ],
    "ghayr-jaifah": [
        "non-penetrating wound", "surface wound", "superficial injury"
    ],
    "ghayr jaifah": [
        "non-penetrating wound", "surface wound", "superficial injury"
    ],

    # ── Organ destruction ─────────────────────────────────────────────────────
    "itlaf-i-salahiyyat-i-udw": [
        "permanent organ impairment", "loss of function of organ",
        "grievous hurt impairing organ"
    ],
    "itlaf-i-udw": [
        "dismemberment", "amputation", "loss of organ",
        "severing of limb", "grievous hurt"
    ],
    "itlaf": ["destruction of body part", "loss of limb or organ"],

    # ── Societal harm ─────────────────────────────────────────────────────────
    "fasad-fil-arz": ["grave societal harm", "mischief on earth", "spreading disorder"],
    "fasad fil arz": ["grave societal harm", "mischief on earth", "spreading disorder"],

    # ── Shajjah sub-categories (specific head/face wound types) ─────────────
    "damighah": [
        "brain membrane wound", "wound reaching brain",
        "most severe head wound"
    ],
    "munaqqilah": [
        "bone-displacing wound", "wound causing bone to shift",
        "severe head fracture"
    ],
    "hashimah": [
        "bone-breaking wound", "fracture wound",
        "head wound breaking bone"
    ],
    "mudihah": [
        "bone-exposing wound", "wound exposing skull",
        "head wound exposing bone"
    ],

    # ── Giving women as settlement / blood money ──────────────────────────────
    "badal-i-sulh": [
        "woman given in marriage as settlement",
        "female as compensation", "giving girl as blood money"
    ],
    "wanni": [
        "giving woman as blood price", "female given as compensation for killing",
        "tribal settlement by marriage"
    ],
    "swara": [
        "giving woman as blood price", "female given as compensation for killing",
        "tribal settlement by marriage"
    ],

    # ── Abortion ──────────────────────────────────────────────────────────────
    "isqat-i-janin": [
        "late-stage abortion", "abortion of formed foetus",
        "causing miscarriage of viable foetus"
    ],
    "isqat-i-hamal": [
        "abortion", "miscarriage", "causing miscarriage",
        "termination of pregnancy"
    ],
    "isqat": ["abortion", "miscarriage"],

    # ── Hurt (legacy term kept for coverage) ──────────────────────────────────
    "zirh": ["hurt", "injury", "bodily harm"],
}

def clean_text(text):
    # Fix 2: preserve newlines — only collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def find_keywords(text, term_map):
    found = []
    seen = set()  # Fix minor: O(1) deduplication
    lower_text = text.lower()
    for term, meanings in term_map.items():
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, lower_text):
            for item in [term] + meanings:
                if item not in seen:
                    seen.add(item)
                    found.append(item)
    return found

def make_normalized_text(text, term_map):
    # Fix 1: single-pass replacement so shorter terms can't re-match
    # inside already-replaced text (e.g. "qatl" firing inside "qatl-i-amd (...)").
    sorted_terms = sorted(term_map.items(), key=lambda x: len(x[0]), reverse=True)

    combined = re.compile(
        "|".join(r"\b" + re.escape(term) + r"\b" for term, _ in sorted_terms),
        re.IGNORECASE,
    )
    lookup = {term.lower(): meanings for term, meanings in sorted_terms}

    def replace(m):
        matched = m.group(0)
        meanings = lookup.get(matched.lower(), [])
        return matched + " (" + ", ".join(meanings) + ")"

    normalized = combined.sub(replace, text)
    normalized = clean_text(normalized)
    return normalized

def main():
    # Fix 3: use __file__-relative paths so script works from any directory.
    # Write back to ppc_sections.json so build_vectorstore_sections.py picks it up.
    input_file  = _SCRIPTS / "../output/ppc_sections.json"
    output_file = _SCRIPTS / "../output/ppc_sections.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for item in data:
        section_id = str(item.get("section_id", "")).strip()
        # support both old {"text"} and new {"original_text"} schemas
        text = (item.get("original_text") or item.get("text", "")).strip()

        normalized_text = make_normalized_text(text, term_map)
        keywords = find_keywords(text, term_map)

        new_item = {
            "section_id":      section_id,
            "text":            text,
            "normalized_text": normalized_text,
            "keywords":        keywords,
        }

        new_data.append(new_item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"done. saved to {output_file}")
    print(f"total sections processed: {len(new_data)}")

if __name__ == "__main__":
    main()