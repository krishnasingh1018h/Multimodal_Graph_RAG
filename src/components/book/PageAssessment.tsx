import { useState } from "react";
import {
  generateQuiz,
  submitQuiz,
  STUDENT_ID,
  type QuizFeedback,
  type QuizQuestion,
} from "../../lib/api";

export function PageAssessment() {
  const [scope, setScope] = useState("");
  const [count, setCount] = useState(5);
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [feedback, setFeedback] = useState<QuizFeedback | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function createQuiz() {
    if (!scope.trim() || busy) return;
    setBusy(true);
    setError("");
    setFeedback(null);
    setAnswers({});
    try {
      const result = await generateQuiz(scope.trim(), count);
      setQuestions(result.questions);
      if (result.questions.length === 0) {
        setError("No questions were generated. Try adding course material or another topic.");
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Quiz generation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function gradeQuiz() {
    if (!scope.trim() || !questions.length || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await submitQuiz(scope.trim(), questions, answers);
      setFeedback(result);
      window.dispatchEvent(new Event("learning-companion:mastery-updated"));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Quiz grading failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="mt-6 grid gap-4 rounded-xl border border-white/10 bg-black/20 p-4 sm:grid-cols-[minmax(0,1fr)_220px] sm:items-end">
        <label className="block text-xs text-warm-cream/65">
          Scope / topic
          <input className="mt-2 w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2.5 text-sm text-warm-cream outline-none focus:border-ember/60" onChange={(event) => setScope(event.target.value)} value={scope} />
        </label>
        <label className="block text-xs text-warm-cream/65">
          <span className="flex justify-between">Questions <span className="font-mono text-ember">{count}</span></span>
          <input className="mt-3 w-full accent-[#E38343]" max={8} min={1} onChange={(event) => setCount(Number(event.target.value))} type="range" value={count} />
          <span className="mt-1 flex justify-between font-mono text-[9px] text-warm-cream/35"><span>1</span><span>8</span></span>
        </label>
        <button className="rounded-lg bg-ember px-4 py-3 text-xs font-semibold text-deep-ink transition hover:brightness-110 disabled:cursor-wait disabled:opacity-60 sm:col-span-2" disabled={busy || !scope.trim()} onClick={createQuiz} type="button">
          {busy ? "Working..." : "Generate Quiz"}
        </button>
      </div>
      {error && <p aria-live="assertive" className="mt-4 text-sm text-rose-300">{error}</p>}
      {questions.length > 0 ? (
        <div className="mt-5 space-y-4">
          {questions.map((question, index) => {
            const result = feedback?.results.find((item) => item.qhash === question.qhash);
            return (
              <article className="rounded-xl border border-white/10 bg-black/20 p-4" key={question.qhash}>
                <p className="font-mono text-[9px] uppercase tracking-wider text-ember">
                  Question {index + 1} · {question.difficulty} · {question.qtype}
                </p>
                <p className="mt-2 text-sm leading-6 text-warm-cream/85">{question.stem}</p>
                {question.options.length > 0 ? (
                  <select
                    aria-label={`Answer to question ${index + 1}`}
                    className="mt-3 w-full rounded-lg border border-white/10 bg-deep-ink px-3 py-2 text-sm text-warm-cream"
                    disabled={Boolean(feedback) || busy}
                    onChange={(event) => setAnswers((current) => ({ ...current, [question.qhash]: event.target.value }))}
                    value={answers[question.qhash] ?? ""}
                  >
                    <option value="">Choose an answer</option>
                    {question.options.map((option) => <option key={option} value={option}>{option}</option>)}
                  </select>
                ) : (
                  <textarea
                    aria-label={`Answer to question ${index + 1}`}
                    className="mt-3 w-full rounded-lg border border-white/10 bg-black/30 px-3 py-2 text-sm text-warm-cream outline-none focus:border-ember/60"
                    disabled={Boolean(feedback) || busy}
                    onChange={(event) => setAnswers((current) => ({ ...current, [question.qhash]: event.target.value }))}
                    placeholder="Write your answer..."
                    rows={2}
                    value={answers[question.qhash] ?? ""}
                  />
                )}
                {result && (
                  <div className="mt-3 border-t border-white/10 pt-3 text-sm">
                    <p className={result.correct ? "text-emerald-300" : "text-rose-300"}>
                      {result.correct ? "Correct" : "Review this one"} · {result.feedback}
                    </p>
                    {!result.correct && <p className="mt-2 text-xs text-warm-cream/60">Answer: {result.key}</p>}
                    {result.source && <p className="mt-2 text-xs text-warm-cream/45">Source: {result.source}</p>}
                  </div>
                )}
              </article>
            );
          })}
          {!feedback && (
            <button className="rounded-lg border border-ember/40 px-4 py-3 text-xs font-semibold text-warm-cream transition hover:bg-ember/10 disabled:opacity-60" disabled={busy} onClick={gradeQuiz} type="button">
              {busy ? "Grading..." : "Submit answers"}
            </button>
          )}
          {feedback && <p className="text-sm text-warm-cream/70">Quiz complete for {STUDENT_ID}. Mastery has been updated.</p>}
        </div>
      ) : (
        <div className="mt-5 rounded-xl border border-white/10 bg-black/20 px-4 py-8 text-center text-xs text-warm-cream/40">
          Enter a topic and generate a quiz to begin.
        </div>
      )}
    </>
  );
}
