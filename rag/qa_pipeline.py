# app/rag/qa_pipeline.py

import re
import time
from app.rag.embeddings import encode_texts


# ==========================================================
# TEXT CLEANING
# ==========================================================
def clean_text(text):
    """Clean and normalize extracted text."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", str(text))
    replacements = {
        "\u00e2\u0080\u0098": "'",
        "\u00e2\u0080\u0099": "'",
        "\u00e2\u0080\u009c": '"',
        "\u00e2\u0080\u009d": '"',
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u0080\u0093": "-",
        "\u00e2\u0080\u0094": "-",
        "\u00c2": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"\b\d+\b(?=\s*$)", "", text)
    return text.strip()


# ==========================================================
# QUERY PREPROCESSING
# ==========================================================
QUERY_PREFIXES = [
    "what is",
    "what are",
    "define",
    "explain",
    "meaning of",
    "tell me about",
    "give me",
    "describe",
]

QUERY_ALIASES = {
    "ai": "artificial intelligence",
    "a.i": "artificial intelligence",
    "a.i.": "artificial intelligence",
    "llm": "large language models llm",
    "llms": "large language models llm",
    "ml": "machine learning ml",
    "mlops": "machine learning operations mlops",
}

REQUIRED_EVIDENCE_PHRASES = {
    "ai": ["artificial intelligence"],
    "a.i": ["artificial intelligence"],
    "a.i.": ["artificial intelligence"],
    "llm": ["llm", "large language model", "large language models"],
    "llms": ["llm", "large language model", "large language models"],
    "ml": ["ml", "machine learning"],
    "mlops": ["mlops", "machine learning operations"],
}


def strip_query_prefixes(query):
    q = query.lower().strip()

    for prefix in QUERY_PREFIXES:
        if q.startswith(prefix):
            return q[len(prefix):].strip()

    return q


def normalize_query(query):
    if not query:
        return ""

    q = strip_query_prefixes(query)
    q = QUERY_ALIASES.get(q, q)

    return q


def strip_document_chrome(text):
    text = clean_text(text)
    if not text:
        return ""

    text = re.sub(
        r"(?i)\bLarge\s+Language\s+Models\s*\(LLM\)\s+Large\s+Language\s+Models\s*\(LLM[’']?s?\)",
        "Large Language Models (LLM)",
        text,
    )
    text = re.sub(r"(?i)\bLarge\s*-\s*Language\s+Models\b", "Large Language Models", text)
    text = re.sub(
        r"(?i)^\s*INTRODUCTION TO AI\s+World Travel\s*&\s*Tourism Council\s*<\s*Contents\s*\|\s*\d+\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^\s*INTRODUCTION TO AI\s+World Travel\s*&\s*Tourism Council\s*\d+\s*<\s*Contents\s*\|\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^\s*World Travel\s*&\s*Tourism Council\s*<\s*Contents\s*\|\s*\d+\s*",
        "",
        text,
    )
    text = re.sub(
        r"(?i)^\s*World Travel\s*&\s*Tourism Council\s*\d+\s*<\s*Contents\s*\|\s*",
        "",
        text,
    )
    text = re.sub(
        r"^\s*[A-Z][A-Z0-9\s&,:;'\"()/-]{18,}\s+(?=(?:One|There|Large|Machine|Artificial|For|When|The|A|An|This|These)\b)",
        "",
        text,
    )
    return clean_text(text)


def trim_answer_to_question(query, answer):
    query = query.lower()
    answer = strip_document_chrome(answer)

    stop_markers = []
    if "large language model" in query or " llm" in f" {query}":
        stop_markers.extend([" Foundational Models ", " Foundation Models "])
    if "machine learning" in query and "operations" not in query:
        stop_markers.extend([" • Deep Learning ", " Deep Learning : "])

    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker, 1)[0].strip()

    return clean_text(answer)


# ==========================================================
# SCORING
# ==========================================================
STOPWORDS = {
    "what", "where", "when", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an",
    "of", "to", "in", "on", "for", "with", "and", "or", "by", "about",
    "tell", "give", "explain", "define", "meaning", "document", "pdf",
}

SHORT_MEANINGFUL_TERMS = {"ai", "ml", "vr", "ar"}


def query_terms(query):
    return [
        word
        for word in re.findall(r"[a-z0-9]+", query.lower())
        if word not in STOPWORDS and (len(word) > 2 or word.isdigit() or word in SHORT_MEANINGFUL_TERMS)
    ]


def text_contains_phrase(text, phrase):
    phrase_terms = re.findall(r"[a-z0-9]+", phrase.lower())
    if not phrase_terms:
        return False

    pattern = r"\b" + r"\W+".join(re.escape(term) for term in phrase_terms) + r"\b"
    return bool(re.search(pattern, text.lower()))


def required_evidence_phrases_for_query(original_query, processed_query):
    raw_query = strip_query_prefixes(original_query)
    phrases = list(REQUIRED_EVIDENCE_PHRASES.get(raw_query, []))

    if raw_query == "machine learning operations" or processed_query.startswith("machine learning operations"):
        phrases.extend(["mlops", "machine learning operations"])

    return list(dict.fromkeys(phrases))


def candidate_has_required_evidence(original_query, processed_query, candidate):
    required_phrases = required_evidence_phrases_for_query(original_query, processed_query)
    if not required_phrases:
        return True

    text = strip_document_chrome(_chunk_text(candidate["chunk"]))
    return any(text_contains_phrase(text, phrase) for phrase in required_phrases)


def keyword_score(query, text):
    query_words = set(query_terms(query))
    text_words = set(re.findall(r"\w+", text.lower()))

    if not query_words:
        return 0

    return len(query_words.intersection(text_words))


def extract_section_query(query):
    match = re.search(r"\b(?:section|sec\.?)\s*([0-9]{1,4}[a-z]?)\b", query.lower())
    return match.group(1).upper() if match else None


def is_definition_query(query):
    return bool(re.search(r"^\s*(what\s+is|what\s+are|define|meaning\s+of|explain)\b", query.lower()))


def answer_looks_like_definition(term, answer):
    if not term:
        return True

    term = term.lower().strip()
    if term in {"artificial intelligence", "ai"}:
        term = "artificial intelligence"

    probe = answer.lower()
    term_variants = [term]

    if term == "large language models llm":
        term_variants.extend(["large language models", "large language model", "llm"])
    elif term == "machine learning ml":
        term_variants.extend(["machine learning", "ml"])
    elif term == "machine learning operations mlops":
        term_variants.extend(["machine learning operations", "mlops"])

    for variant in dict.fromkeys(term_variants):
        escaped = re.escape(variant)
        patterns = [
            rf"\b{escaped}\b\s*(?:\([^)]+\))?\s+(?:is|are|means|refers\s+to|describes|denotes)\b",
            rf"\b{escaped}\b\s*(?:\([^)]+\))?\s+(?:systems?|models?|methods?|techniques?)\s+(?:can|are|is|learn|use)\b",
            rf"\bdefined\s+as\b",
            rf"\bdefinition\b",
        ]
        if any(re.search(pattern, probe) for pattern in patterns):
            return True

    return False


BAD_SECTION_HEADING_PREFIXES = {
    "added", "subs", "sub", "ins", "inserted", "rep", "repealed", "omitted",
    "chapter", "w", "vide", "ibid", "cl", "sch", "section",
}

LEGAL_SECTION_BODY_RE = re.compile(
    r"\b(whoever|shall|punished|imprisonment|fine|offence|offense|"
    r"cognizable|non-cognizable|bailable|non-bailable|court|magistrate)\b",
    re.I,
)

STRONG_SECTION_BODY_RE = re.compile(
    r"\b(whoever|shall|provided\s+that|explanation|illustration)\b",
    re.I,
)


def is_contents_like_text(text):
    probe = clean_text(text)[:2200]
    if not probe:
        return False

    lower_probe = probe.lower()
    has_contents_label = bool(re.search(r"\b(contents|table\s+of\s+contents|arrangement\s+of\s+sections|index)\b", lower_probe))
    legal_body = bool(STRONG_SECTION_BODY_RE.search(probe))
    numbered_heading_count = len(re.findall(
        r"(?<!\w)(?:section\s+|sec\.?\s*)?[0-9]{1,4}[A-Z]?\s*[\.\-:)\]]\s+[A-Z][A-Za-z][^.;]{5,90}",
        probe,
    ))
    chapter_count = len(re.findall(r"\bchapter\s+[ivxlcdm0-9]+\b", lower_probe))

    return not legal_body and (
        has_contents_label
        or numbered_heading_count >= 4
        or (chapter_count >= 2 and numbered_heading_count >= 2)
    )


def section_body_present_after(text, start):
    return bool(STRONG_SECTION_BODY_RE.search(text[start:start + 1400]))


def _looks_like_amendment_note(text_after_number):
    words = re.findall(r"[A-Za-z]+", text_after_number[:80].lower())
    if not words:
        return True
    if words[0] == "act":
        return bool(re.match(r"\s*act\s+[0-9]", text_after_number, re.I))
    return words[0] in BAD_SECTION_HEADING_PREFIXES


def section_heading_score(section_id, text):
    if not section_id:
        return 0

    if is_contents_like_text(text):
        return 0

    escaped = re.escape(section_id)
    contents_like = bool(re.search(r"\b(contents|table\s+of\s+contents)\b", text[:500], re.I))
    many_numbered_headings = len(re.findall(r"(?m)^\s*[0-9]{1,4}[A-Z]?\s*[\.\-:)\]]\s+", text)) >= 3
    legalish_context = bool(re.search(r"\b(act|code|shall|whoever|punishment|offence|section)\b", text, re.I))
    heading_patterns = [
        (130, rf"(?im)^\s*(?:section\s+|sec\.?\s*){escaped}\s*[\.\-:)\]]?\s*(?P<title>.{{0,120}})"),
    ]

    if legalish_context and not (contents_like and many_numbered_headings):
        heading_patterns.extend([
            (120, rf"(?im)^\s*{escaped}\s*[\.\-:)\]]\s+(?P<title>.{{1,120}})"),
            (115, rf"(?i)(?:^|[\r\n]|[.;]\s+){escaped}\s*[\.\-:)\]]\s+(?P<title>[A-Z][^.:\n]{{1,140}})"),
        ])

    for weight, pattern in heading_patterns:
        for match in re.finditer(pattern, text):
            if (
                not _looks_like_amendment_note(match.group("title"))
                and section_body_present_after(text, match.end())
            ):
                return weight

    inline_patterns = [
        (45, rf"(?i)\bsection\s+{escaped}\b"),
        (40, rf"(?i)\bsec\.?\s*{escaped}\b"),
    ]

    for weight, pattern in inline_patterns:
        if re.search(pattern, text):
            return weight

    return 0


def lexical_score(query, text):
    clean = text.lower()
    score = keyword_score(query, text) * 10
    normalized = normalize_query(query)

    if normalized and normalized in clean:
        score += 35

    for term in query_terms(query):
        if re.search(rf"\b{re.escape(term)}\b", clean):
            score += 3

    return score


def retrieval_is_strong_enough(query, candidate, answer):
    if not answer or answer == "I could not find the answer in the document.":
        return False

    raw_text = _chunk_text(candidate["chunk"])
    overlap = keyword_score(query, raw_text)
    normalized = normalize_query(query)
    phrase_match = bool(normalized and normalized in raw_text.lower())
    vector_support = candidate.get("distance") is not None and candidate.get("distance", 999999.0) < 1.35

    if overlap >= 2:
        return True
    if overlap >= 1 and (phrase_match or vector_support):
        return True
    return False


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def extract_relevant_sentences(query, text, max_sentences=4):
    sentences = split_into_sentences(text)
    scored_sentences = []

    for sentence in sentences:
        score = keyword_score(query, sentence)
        if score > 0:
            scored_sentences.append((score, sentence))

    if not scored_sentences:
        return ""

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = []

    for _, sentence in scored_sentences:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break

    return strip_document_chrome(" ".join(selected))


def next_section_lookahead(section_id):
    if section_id and section_id.isdigit():
        next_section = int(section_id) + 1
        return rf"(?=\s+(?:section\s+|sec\.?\s*)?{next_section}[A-Z]?\s*[\.\-:)\]]\s+[A-Z]|\Z)"

    return r"(?=\s+(?:section\s+|sec\.?\s*)?[0-9]{1,4}[A-Z]?\s*[\.\-:)\]]\s+[A-Z]|\Z)"


def section_answer_has_body(answer):
    return bool(answer and len(clean_text(answer)) >= 80 and STRONG_SECTION_BODY_RE.search(answer))


def extract_section_answer(section_id, text):
    if is_contents_like_text(text):
        return ""

    escaped = re.escape(section_id)
    boundary = next_section_lookahead(section_id)
    legalish_context = bool(re.search(r"\b(act|code|shall|whoever|punishment|offence|section)\b", text, re.I))
    patterns = [
        rf"(?ims)((?:section\s+|sec\.?\s*){escaped}\b(?P<title>.{{0,120}}).*?){boundary}",
    ]
    if legalish_context:
        patterns = [
            rf"(?ims)(^\s*{escaped}\s*[\.\-:)\]]\s+(?P<title>.{{1,120}}).*?){boundary}",
            rf"(?ims)((?:^|[\r\n]|[.;]\s+){escaped}\s*[\.\-:)\]]\s+(?P<title>[A-Z].{{1,160}}).*?){boundary}",
            *patterns,
        ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            if _looks_like_amendment_note(match.group("title")):
                continue
            answer = strip_document_chrome(match.group(1))
            if not section_answer_has_body(answer):
                continue
            if len(answer) > 1200:
                answer = answer[:1200] + "..."
            return answer

    return ""


def _chunk_text(chunk):
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return str(chunk)


def _chunk_page(chunk):
    if isinstance(chunk, dict):
        return chunk.get("page")
    return None


def _chunk_index(chunk):
    if isinstance(chunk, dict):
        return chunk.get("chunk_index", 0)
    return 0


def ordered_chunks(chunks):
    return sorted(
        enumerate(chunks),
        key=lambda item: (
            _chunk_page(item[1]) or 0,
            _chunk_index(item[1]) or item[0],
            item[0],
        ),
    )


def section_candidate_context(ordered, start_position, window=3):
    parts = []
    for _, chunk in ordered[start_position:start_position + window]:
        text = _chunk_text(chunk).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def find_exact_section_candidate(section_id, chunks):
    ordered = ordered_chunks(chunks)
    matches = []

    for position, (original_index, chunk) in enumerate(ordered):
        first_chunk_score = section_heading_score(section_id, _chunk_text(chunk))
        if first_chunk_score < 100:
            continue

        context = section_candidate_context(ordered, position)
        answer = extract_section_answer(section_id, context)
        if not answer:
            continue

        score = section_heading_score(section_id, context)
        if score < 100:
            continue

        matches.append({
            "score": score + 100,
            "distance": None,
            "chunk": {
                "text": context,
                "page": _chunk_page(chunk),
                "chunk_index": _chunk_index(chunk),
            },
            "chunk_index": original_index,
            "answer": answer,
        })

    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            item["score"],
            -(_chunk_page(item["chunk"]) or 999999),
            -(_chunk_index(item["chunk"]) or 999999),
        ),
        reverse=True,
    )
    return matches[0]


def section_result(section_id, chunks, started):
    candidate = find_exact_section_candidate(section_id, chunks)
    if not candidate:
        return {
            "answer": f"I could not find Section {section_id} in the document.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": round(time.time() - started, 2),
        }

    answer = candidate["answer"]
    source = _source_from_candidate(candidate, 1, section_id=section_id)
    pages = [source["page"]] if source.get("page") else []

    return {
        "answer": answer,
        "sources": [source],
        "pages": pages,
        "confidence": "high",
        "answer_time": round(time.time() - started, 2),
    }


def _source_from_candidate(candidate, source_id, query=None, section_id=None):
    chunk = candidate["chunk"]
    text = strip_document_chrome(_chunk_text(chunk))
    page = chunk.get("page") if isinstance(chunk, dict) else None
    chunk_index = chunk.get("chunk_index") if isinstance(chunk, dict) else candidate.get("chunk_index")

    if section_id:
        focused = extract_section_answer(section_id, text)
    elif query:
        focused = trim_answer_to_question(query, extract_relevant_sentences(query, text, max_sentences=2))
    else:
        focused = ""

    snippet_source = focused if len(focused) >= 40 else text
    snippet = snippet_source[:360]
    if len(snippet_source) > 360:
        snippet += "..."

    return {
        "id": source_id,
        "page": page,
        "chunk_index": chunk_index,
        "score": int(candidate.get("score", 0)),
        "distance": candidate.get("distance"),
        "snippet": snippet,
    }


# ==========================================================
# MAIN RAG FUNCTIONS
# ==========================================================
def generate_rag_result(query, chunks, index, top_k=10):
    """Return an answer plus source citations for UI/API/export use."""
    started = time.time()

    if not query or not query.strip():
        return {
            "answer": "Please enter a question.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": 0.0,
        }

    if not chunks:
        return {
            "answer": "Please upload and process a document first.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": 0.0,
        }

    original_query = query.strip()
    processed_query = normalize_query(original_query)
    section_id = extract_section_query(original_query)

    if section_id:
        return section_result(section_id, chunks, started)

    if index is None:
        return {
            "answer": "Please upload and process a document first.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": 0.0,
        }

    query_embedding = encode_texts([processed_query]).astype("float32")
    search_k = min(max(top_k, 1), len(chunks))
    distances, indices = index.search(query_embedding, search_k)

    candidates_by_index = {}
    for distance, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            raw_text = _chunk_text(chunk)
            text = clean_text(raw_text)
            if len(text) < 50:
                continue

            score = lexical_score(processed_query, raw_text)
            score += section_heading_score(section_id, raw_text)
            candidates_by_index[int(idx)] = {
                "score": score,
                "distance": float(distance),
                "chunk": chunk,
                "chunk_index": int(idx),
            }

    for idx, chunk in enumerate(chunks):
        raw_text = _chunk_text(chunk)
        text = clean_text(raw_text)
        if len(text) < 50:
            continue

        score = lexical_score(processed_query, raw_text)
        score += section_heading_score(section_id, raw_text)

        if score <= 0:
            continue

        existing = candidates_by_index.get(idx)
        if existing:
            existing["score"] = max(existing["score"], score)
        else:
            candidates_by_index[idx] = {
                "score": score,
                "distance": None,
                "chunk": chunk,
                "chunk_index": idx,
            }

    candidates = list(candidates_by_index.values())

    if not candidates:
        return {
            "answer": "I could not find the answer in the document.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": round(time.time() - started, 2),
        }

    if section_id:
        exact_section_candidates = [
            candidate
            for candidate in candidates
            if section_heading_score(section_id, _chunk_text(candidate["chunk"])) >= 100
        ]
        if not exact_section_candidates:
            return {
                "answer": f"I could not find Section {section_id} in the document.",
                "sources": [],
                "pages": [],
                "confidence": "none",
                "answer_time": round(time.time() - started, 2),
            }
        candidates = exact_section_candidates

    candidates = [
        candidate
        for candidate in candidates
        if candidate_has_required_evidence(original_query, processed_query, candidate)
    ]

    if not candidates:
        return {
            "answer": "I could not find the answer in the document.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": round(time.time() - started, 2),
        }

    candidates.sort(
        key=lambda item: (item["score"], -(item["distance"] or 999999.0)),
        reverse=True,
    )
    best_candidate = candidates[0]

    if best_candidate["score"] <= 0:
        return {
            "answer": "I could not find the answer in the document.",
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": round(time.time() - started, 2),
        }

    if section_id:
        answer = extract_section_answer(section_id, _chunk_text(best_candidate["chunk"]))
    else:
        answer = extract_relevant_sentences(
            processed_query,
            strip_document_chrome(_chunk_text(best_candidate["chunk"])),
            max_sentences=4,
        )
    answer = trim_answer_to_question(processed_query, answer)

    if len(answer) > 1200:
        answer = answer[:1200] + "..."

    if len(answer) < 20:
        answer = "I could not find the answer in the document."

    if not section_id and not retrieval_is_strong_enough(processed_query, best_candidate, answer):
        answer = "I could not find the answer in the document."

    if (
        not section_id
        and answer != "I could not find the answer in the document."
        and is_definition_query(original_query)
        and not answer_looks_like_definition(processed_query, answer)
    ):
        answer = (
            f"The document does not give a clean definition of {processed_query}. "
            f"The closest cited passage says: {answer}"
        )

    if answer == "I could not find the answer in the document.":
        return {
            "answer": answer,
            "sources": [],
            "pages": [],
            "confidence": "none",
            "answer_time": round(time.time() - started, 2),
        }

    if section_id or is_definition_query(original_query):
        meaningful_sources = [best_candidate]
    else:
        meaningful_sources = [candidate for candidate in candidates if candidate["score"] > 0][:4]
    sources = [
        _source_from_candidate(candidate, i + 1, processed_query, section_id)
        for i, candidate in enumerate(meaningful_sources)
    ]
    pages = sorted({source["page"] for source in sources if source.get("page")})

    confidence = "medium"
    if sources and best_candidate["score"] >= 30:
        confidence = "high"
    if not pages:
        confidence = "medium"
    if answer.startswith("The document does not give a clean definition"):
        confidence = "medium"

    return {
        "answer": answer,
        "sources": sources,
        "pages": pages,
        "confidence": confidence,
        "answer_time": round(time.time() - started, 2),
    }


def generate_rag_response(query, chunks, index, top_k=10):
    """Backward-compatible string answer wrapper."""
    return generate_rag_result(query, chunks, index, top_k=top_k)["answer"]
