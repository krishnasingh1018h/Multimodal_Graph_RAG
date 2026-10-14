"""G1 TutorRAG: router -> hybrid_retrieve -> grade -> graph_expand -> generate -> verify -> answer|refuse."""
from __future__ import annotations
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from src.config.settings import get_settings


class TutorState(TypedDict):
    student_id: str
    question: str
    topic_filter: Optional[str]
    docs: list
    context: str
    citations: list
    answer: str
    verdict: Literal["answer", "refuse"]


def n_retrieve(s: TutorState) -> TutorState:
    from src.stores.vector import hybrid_search
    st = get_settings()
    docs = hybrid_search(s["question"], k=st.TOP_K, topic=s.get("topic_filter"))
    s["docs"] = docs
    return s


def n_grade(s: TutorState) -> TutorState:
    """Keyword-overlap gate (threshold from settings); refuse only when truly no overlap."""
    import re
    q = set(re.findall(r"\w+", s["question"].lower()))
    q = {w for w in q if len(w) > 2}
    scored = []
    for d in s["docs"]:
        toks = set(re.findall(r"\w+", d.page_content.lower()))
        overlap = len(q & toks) / max(1, len(q))
        scored.append((overlap, d))
    scored.sort(key=lambda x: -x[0])
    best = scored[0][0] if scored else 0.0
    s["docs"] = [d for overlap, d in scored if overlap > 0]
    s["verdict"] = "answer" if best >= get_settings().GRADE_THRESHOLD else "refuse"
    return s


def n_expand(s: TutorState) -> TutorState:
    from src.stores.graph import expand_prereqs
    if s["verdict"] == "answer":
        concepts = [d.metadata.get("concept", "") for d in s["docs"] if hasattr(d, "metadata")]
        extra = expand_prereqs([c for c in concepts if c])
        if extra:
            s["question"] = s["question"] + f" (prereqs: {', '.join(extra[:3])})"
    return s


def n_generate(s: TutorState) -> TutorState:
    from src.stores.learner import get_mastery
    if s["verdict"] == "refuse":
        s["answer"] = ("I can't answer this from the uploaded course material. "
                       "The sources I searched don't cover it. Try a topic from the course outline.")
        s["citations"] = []
        s["context"] = ""
        return s
    ctx, cites = [], []
    for d in s["docs"][:6]:
        m = d.metadata
        if m.get("page"): cite = f"[{m.get('source_file')}, p.{m['page']}]"
        elif m.get("slide"): cite = f"[{m.get('source_file')}, slide {m['slide']}]"
        elif m.get("t_start") is not None:
            mm, ss = divmod(int(m["t_start"]), 60); cite = f"[{m.get('source_file')}, {mm:02d}:{ss:02d}]"
        else: cite = f"[{m.get('source_file', 'source')}]"
        ctx.append(f"{cite} {d.page_content[:1200]}"); cites.append(cite)
    s["context"] = "\n".join(ctx); s["citations"] = cites
    try:
        from src.chains.prompts import get_answer_chain, _content
        ans = _content(get_answer_chain().invoke(
            {"context": s["context"], "question": s["question"]}))
        if not ans.strip():
            raise ValueError("empty LLM answer")
        s["answer"] = ans
        if not any(c.split(",")[0][1:] in s["answer"] for c in cites):
            s["answer"] += "\n\nSources: " + "; ".join(cites)
    except Exception:
        topic = (s["docs"][0].metadata.get("topic", "this topic") if s["docs"] else "this topic")
        mastery = get_mastery(s.get("student_id", "anon"), topic)
        level = "Simply put: " if mastery < 0.4 else ("In depth: " if mastery > 0.75 else "")
        s["answer"] = (f"{level}Based on your material:\n\n" + "\n".join(ctx[:3])
                       + f"\n\nSources: {'; '.join(cites)}")
    return s


def n_verify(s: TutorState) -> TutorState:
    if s["verdict"] == "refuse" or not s.get("context"):
        return s
    try:
        from src.chains.prompts import get_verify_chain, _content
        v = _content(get_verify_chain().invoke(
            {"context": s["context"][:4000], "answer": s["answer"][:2000]}))
        if "NOT_SUPPORTED" in v.upper():
            s["verdict"] = "refuse"
            s["answer"] = ("This question isn't covered by the uploaded material "
                           "(verification failed), so I'm declining rather than guessing.")
    except Exception:
        pass
    return s


def build_tutor_graph(checkpoint=True):
    g = StateGraph(TutorState)
    for n, fn in [("retrieve", n_retrieve), ("grade", n_grade), ("expand", n_expand),
                  ("generate", n_generate), ("verify", n_verify)]:
        g.add_node(n, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade"); g.add_edge("grade", "expand")
    g.add_edge("expand", "generate"); g.add_edge("generate", "verify"); g.add_edge("verify", END)
    if checkpoint:
        import os
        os.makedirs("data/checkpoints", exist_ok=True)
        conn = sqlite3.connect(get_settings().CHECKPOINT_DB, check_same_thread=False)
        return g.compile(checkpointer=SqliteSaver(conn))
    return g.compile()


def ask(question: str, student_id: str = "anon", topic: str | None = None) -> dict:
    app = build_tutor_graph()
    return app.invoke({"student_id": student_id, "question": question, "topic_filter": topic,
                       "docs": [], "context": "", "citations": [], "answer": "", "verdict": "answer"},
                      config={"configurable": {"thread_id": student_id}})
