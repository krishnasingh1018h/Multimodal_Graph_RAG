"""G2 QuizGraph: scope -> retrieve -> generate -> verify -> dedup -> feedback -> BKT -> report."""
from __future__ import annotations
import hashlib, json, os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from src.config.settings import get_settings
from src.schemas.models import QuizQuestion


class QuizState(TypedDict):
    student_id: str
    scope: str
    difficulty_target: str
    n_questions: int
    questions: list
    results: list
    weak_topics: list


def n_scope(s: QuizState) -> QuizState:
    from src.stores.learner import difficulty_for
    if s["difficulty_target"] == "auto":
        s["difficulty_target"] = difficulty_for(s["student_id"], s["scope"])
    return s


def n_generate(s: QuizState) -> QuizState:
    from src.stores.vector import hybrid_search
    docs = hybrid_search(f"{s['scope']} {s['difficulty_target']} questions", k=6, topic=None)
    ctx = "\n".join(d.page_content[:800] for d in docs)
    src = docs[0].metadata.get("source_file", "") if docs else ""
    used_llm = False
    try:
        from src.chains.prompts import get_quiz_chain, _content
        out = _content(get_quiz_chain().invoke(
            {"n": s["n_questions"], "types": "MCQ/short/numerical",
             "scope": s["scope"], "difficulty": s["difficulty_target"],
             "context": ctx}))
        cand = _parse_quiz_text(out, scope=s["scope"], difficulty=s["difficulty_target"],
                                source=src, n=s["n_questions"])
        if not cand:
            if not out.strip():
                raise ValueError("empty LLM quiz output")  # -> offline fallback
            cand = [QuizQuestion(qtype="short", stem=out[:500], answer_key="see context",
                                 topic=s["scope"], source=src, difficulty=s["difficulty_target"],
                                 explanation="Generated from retrieved context.")]
        used_llm = True
    except Exception:
        # offline fallback: one question per DEFINITIONAL sentence (filtered).
        # Raw chapter text contains URLs, fragments, mid-paragraph continuations
        # ("Therefore, ...", "com: This Q&A site...") — never quiz those.
        cand = []
        import random, re
        random.shuffle(docs)
        for i, d in enumerate(docs[:s["n_questions"] * 4]):
            if len(cand) >= s["n_questions"]:
                break
            for sent in _quiz_worthy_sentences(d.page_content):
                m = d.metadata
                link = f"{m.get('source_file','')}" + (f"#page={m['page']}" if m.get("page") else "")
                concept = _extract_concept(sent)
                cand.append(QuizQuestion(
                    qtype="short",
                    stem=f"Q{len(cand)+1} ({s['difficulty_target']}). {concept}: explain in your own words.",
                    answer_key=sent, topic=d.metadata.get("topic", s["scope"]),
                    source=link, difficulty=s["difficulty_target"],
                    explanation=f"Key point: {sent[:200]} (See {link})"))
                if len(cand) >= s["n_questions"]:
                    break
        if not cand:
            # strict filter found nothing quiz-worthy: relax once (allow longer
            # subjects) rather than returning zero questions
            for i, d in enumerate(docs[:s["n_questions"] * 2]):
                sents = [x.strip() for x in d.page_content.split(".") if 60 <= len(x.strip()) <= 300]
                if not sents:
                    continue
                sent = " ".join(sents[i % len(sents)].split())[:300]
                m = d.metadata
                link = f"{m.get('source_file','')}" + (f"#page={m['page']}" if m.get("page") else "")
                cand.append(QuizQuestion(
                    qtype="short",
                    stem=f"Q{len(cand)+1} ({s['difficulty_target']}). Explain this point in your own words: \"{sent[:120]}...\"?",
                    answer_key=sent, topic=d.metadata.get("topic", s["scope"]),
                    source=link, difficulty=s["difficulty_target"],
                    explanation=f"Key point: {sent[:200]} (See {link})"))
                if len(cand) >= s["n_questions"]:
                    break
    # LLM path is authoritative: keep whatever it produced even if hash-seen
    # (dedup bank only gates the deterministic offline fallback).
    seen = set() if used_llm else _load_hashes()
    fresh = []
    for q in cand:
        h = hashlib.sha256(q.stem.strip().lower().encode()).hexdigest()[:12]
        q.qhash = h
        if h not in seen:
            seen.add(h); fresh.append(q)
    if not used_llm:
        _save_hashes(seen)
    s["questions"] = [q.model_dump() for q in fresh]
    return s


def _parse_quiz_text(out: str, scope: str, difficulty: str, source: str, n: int) -> list:
    """Parse 'Q: ... A: ...' (possibly numbered) blocks into QuizQuestions."""
    import re
    cand = []
    blocks = re.split(r"(?m)^\s*(?:\d+[.)]\s*)?Q\s*:", out)
    for b in blocks[1:]:
        parts = re.split(r"(?m)^\s*A\s*:", b, maxsplit=1)
        stem = parts[0].strip()[:400]
        key = parts[1].strip().splitlines()[0][:300] if len(parts) > 1 else "see context"
        if len(stem) < 10:
            continue
        cand.append(QuizQuestion(qtype="short", stem=stem, answer_key=key, topic=scope,
                                 source=source, difficulty=difficulty,
                                 explanation=f"See {source}"))
        if len(cand) >= n:
            break
    return cand


def _quiz_worthy_sentences(text: str) -> list[str]:
    """Yield sentences that make sane quiz prompts. Rejects URLs, fragments,
    discourse continuations (Therefore/Then/However...), questions, code."""
    import re
    _bad_start = ("therefore", "then", "however", "moreover", "furthermore", "also",
                  "but ", "and ", "or ", "so ", "thus", "hence", "meanwhile", "next,",
                  "finally,", "first,", "second,", "third,", "for example", "for instance",
                  "in addition", "on the other hand", "as a result", "in contrast",
                  "alternatively", "nevertheless", "nonetheless", "instead",
                  "com:", "http", "www.", "see figure", "as shown", "note that",
                  "it is important", "keep in mind", "getting started")
    _definitional = (" is ", " are ", " refers to", " means ", " involves ",
                     " consists of", " defines ", " describes ", " enables ",
                     " allows ", " provides ", " uses ", " relies on")
    _bad_subj = ("it", "this", "that", "these", "those", "they", "them",
                 "he", "she", "we", "you", "if", "when", "while", "the",
                 "chapter", "section", "getting", "alternatively", "maybe")
    out = []
    for raw in re.split(r"(?<=[.!?])\s+", text):
        s = " ".join(raw.split())
        low = s.lower()
        if not (60 <= len(s) <= 300):
            continue
        if any(tok in low for tok in ("http", "www.", "#", ".com", ".org", "freenode",
                                      "irc", "://", "```", ">>>", "import ",
                                      "def ", "    ", "\t", "[", "]")):
            continue
        if low.startswith(_bad_start) or s.endswith(("?", ":", ";")):
            continue
        if not any(pat in low for pat in _definitional):
            continue
        # reject sentences whose SUBJECT is a pronoun/chapter-ref — the extracted
        # concept would be meaningless ("The complexity...", "Chapter 11...")
        import re as _re
        subj = _re.split(r"\s+(?:is|are|refers to|means|involves|consists of)\s+",
                         s, maxsplit=1, flags=_re.I)[0].lower()
        words = subj.split()
        if not words or words[0] in _bad_subj or len(words) > 6:
            continue
        out.append(s)
    return out


def _extract_concept(sent: str) -> str:
    """Short concept label = subject before 'is/are/refers to' (<=8 words)."""
    import re
    m = re.split(r"\s+(?:is|are|refers to|means|involves|consists of)\s+",
                 sent, maxsplit=1, flags=re.I)
    subj = m[0].strip().rstrip(",;:") if m else sent
    words = subj.split()
    return " ".join(words[:8]) if words else "This concept"


def _load_hashes() -> set:
    p = get_settings().QUESTION_BANK
    if os.path.exists(p):
        try:
            return {json.loads(l).get("qhash") for l in open(p) if l.strip()}
        except Exception:
            return set()
    return set()


def _save_hashes(seen: set):
    # deprecated: hashes persist in n_feedback (per graded question). Kept for compat.
    return


def reset_bank() -> None:
    """Clear the dedup bank (used by tests/simulator for isolation)."""
    try:
        if os.path.exists(get_settings().QUESTION_BANK):
            os.remove(get_settings().QUESTION_BANK)
    except Exception:
        pass


def n_verify(s: QuizState) -> QuizState:
    """Blind re-answer check for short/numerical; MCQ key sanity; sympy for numerical.
    Drops questions the verifier marks NOT_SUPPORTED (unless verifier errors -> keep)."""
    try:
        from src.chains.prompts import get_verify_chain, _content
        chain = get_verify_chain()
        keep = []
        for q in s["questions"]:
            if q.get("qtype") == "numerical":
                try:
                    import sympy
                    sympy.sympify(q["answer_key"])  # must parse
                    keep.append(q)
                except Exception:
                    q["verify_note"] = "dropped: answer_key not parseable"
                continue
            try:
                v = _content(chain.invoke(
                    {"context": (q.get("explanation", "") + " | source: "
                                 + q.get("source", ""))[:500],
                     "answer": q["answer_key"][:500]}))
                if "NOT_SUPPORTED" in (v or "").upper():
                    q["verify_note"] = f"dropped: verifier said NOT_SUPPORTED ({v[:120]})"
                    continue
            except Exception:
                pass  # verifier down -> keep (offline-safe)
            keep.append(q)
        s["questions"] = keep
    except Exception:
        pass
    if not s["questions"]:
        from src.stores.vector import hybrid_search

        docs = hybrid_search(
            f"{s['scope']} {s['difficulty_target']} questions", k=6, topic=None)
        fallback = []
        seen_questions: set[str] = set()
        for doc in docs:
            for sentence in _quiz_worthy_sentences(doc.page_content):
                metadata = doc.metadata
                source = metadata.get("source_file", "")
                if metadata.get("page"):
                    source += f"#page={metadata['page']}"
                concept = _extract_concept(sentence)
                question = QuizQuestion(
                    qtype="short",
                    stem=f"Explain {concept} in your own words.",
                    answer_key=sentence,
                    topic=metadata.get("topic", s["scope"]),
                    source=source,
                    difficulty=s["difficulty_target"],
                    explanation=f"Key point: {sentence[:200]} (See {source})",
                )
                question_hash = hashlib.sha256(
                    question.stem.strip().lower().encode()).hexdigest()[:12]
                if question_hash in seen_questions:
                    continue
                seen_questions.add(question_hash)
                question.qhash = question_hash
                fallback.append(question)
                if len(fallback) >= s["n_questions"]:
                    break
            if len(fallback) >= s["n_questions"]:
                break
        s["questions"] = [question.model_dump() for question in fallback]
    return s


def n_feedback(s: QuizState, answers: dict | None = None) -> QuizState:
    """Grade + BKT update + weak-topic report. Pass answers={qhash: student_ans}.

    Grading (no LLM needed, fully verifiable):
      1. exact/overlap: your answer contained in the key sentence (or vice versa)
      2. keyword check: >=60% of the key's content words (len>3) appear in yours
    Each result carries the key sentence + source page so you can verify by
    opening the cited page and comparing. LLM semantic grading is layered on
    top when available (ornith), but never overrides a keyword pass.
    """
    from src.stores.learner import update
    from collections import Counter
    import re
    res, wrong = [], Counter()
    bank_lines = []
    for q in s["questions"]:
        given = (answers or {}).get(q["qhash"], "")
        topic = q.get("topic") or s["scope"]
        if topic == "general":
            topic = s["scope"]
        key = q["answer_key"]
        ok, method = _grade_short(given, key)
        if not ok and given.strip():
            # LLM semantic second opinion (e.g. paraphrase the key accepts)
            verdict = _llm_grade(given, key, q.get("source", ""))
            if verdict:
                ok, method = True, "llm-semantic"
        update(s["student_id"], topic, bool(ok))
        if not ok:
            wrong[topic] += 1
        res.append({"qhash": q["qhash"], "correct": ok, "method": method,
                    "given": given, "key": key, "source": q.get("source", ""),
                    "verify": (f"Open {q.get('source','')} and check the key sentence. "
                               f"Keyword overlap graded ({method})."),
                    "feedback": f"{'Correct.' if ok else 'Not quite — review ' + q.get('source','') + '. '} {q.get('explanation','')}"})
        bank_lines.append(json.dumps({"qhash": q["qhash"], "stem": q["stem"]}))
    s["results"] = res
    s["weak_topics"] = [t for t, _ in wrong.most_common(3)]
    try:
        os.makedirs(os.path.dirname(get_settings().QUESTION_BANK), exist_ok=True)
        open(get_settings().QUESTION_BANK, "a").write("\n".join(bank_lines) + ("\n" if bank_lines else ""))
    except Exception:
        pass
    return s


def _grade_short(given: str, key: str) -> tuple[bool, str]:
    """Deterministic short-answer grading. Returns (correct, method)."""
    import re
    g, k = (given or "").strip().lower(), (key or "").strip().lower()
    if not g:
        return False, "blank"
    if g in k or k in g:
        return True, "exact-overlap"
    stop = {"the", "and", "for", "with", "that", "this", "from", "into",
            "your", "own", "words", "explain"}
    kw = [w for w in re.findall(r"\w+", k) if len(w) > 3 and w not in stop]
    if not kw:
        return False, "no-keywords"
    hit = sum(1 for w in kw if w in g)
    return (hit / len(kw) >= 0.6), f"keyword-{hit}/{len(kw)}"


def _llm_grade(given: str, key: str, source: str) -> bool:
    """Semantic second opinion via verifier LLM. True=accept, False/None=reject.
    Never raises (offline-safe)."""
    try:
        from src.chains.prompts import get_verify_chain, _content
        v = _content(get_verify_chain().invoke(
            {"context": f"Expected answer (from {source}): {key}",
             "answer": f"Student answer: {given}. "
                       "Is the student answer semantically equivalent?"}))
        return "SUPPORTED" in (v or "").upper() and "NOT_SUPPORTED" not in (v or "").upper()
    except Exception:
        return False


def build_quiz_graph():
    g = StateGraph(QuizState)
    g.add_node("scope", n_scope); g.add_node("generate", n_generate); g.add_node("verify", n_verify)
    g.set_entry_point("scope")
    g.add_edge("scope", "generate"); g.add_edge("generate", "verify"); g.add_edge("verify", END)
    return g.compile()
