import { useState, type FormEvent } from "react";
import { askTutor, type Citation } from "../../lib/api";

interface Message {
  role: "student" | "companion";
  text: string;
  citations?: string[];
  verdict?: Citation["verdict"];
}

export function PageTutor() {
  const [topic, setTopic] = useState("");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function askQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || busy) return;
    setMessages((current) => [...current, { role: "student", text: prompt }]);
    setQuestion("");
    setBusy(true);
    setError("");
    try {
      const result = await askTutor(prompt, topic.trim() || undefined);
      setMessages((current) => [
        ...current,
        {
          role: "companion",
          text: result.answer,
          citations: result.citations,
          verdict: result.verdict,
        },
      ]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "The request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <p className="mb-5 text-sm text-warm-cream/60">Your conversation will appear here.</p>
      <div aria-live="polite" className="mt-6 max-h-64 min-h-28 space-y-3 overflow-y-auto rounded-xl border border-white/10 bg-black/20 p-3 pr-1">
        {messages.length > 0 ? messages.map((message, index) => (
          <div className={`max-w-[92%] rounded-2xl border p-4 ${message.role === "student" ? "ml-auto border-ember/20 bg-ember/[0.09]" : "mr-auto border-white/10 bg-black/25"}`} key={`${message.role}-${index}`}>
            <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.18em] text-ember">
              {message.role === "student" ? "You · question" : `Tutor · ${message.verdict ?? "answer"}`}
            </p>
            <p className="text-sm leading-6 text-warm-cream/85">{message.text}</p>
            {message.citations && message.citations.length > 0 && (
              <p className="mt-3 text-xs text-warm-cream/55">Sources: {message.citations.join("; ")}</p>
            )}
          </div>
        )) : (
          <p className="flex min-h-20 items-center justify-center text-xs text-warm-cream/40">No messages yet.</p>
        )}
      </div>
      <form className="mt-5 grid gap-3 border-t border-white/10 pt-4 sm:grid-cols-[minmax(0,1fr)_auto]" onSubmit={askQuestion}>
        <div className="grid gap-2">
          <input className="w-full rounded-xl border border-white/10 bg-black/25 px-4 py-3 text-sm text-warm-cream outline-none placeholder:text-warm-cream/35 focus:border-ember/60" onChange={(event) => setQuestion(event.target.value)} placeholder="Ask from your material..." value={question} />
          <input className="w-full rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-xs text-warm-cream/80 outline-none placeholder:text-warm-cream/35 focus:border-ember/50" onChange={(event) => setTopic(event.target.value)} placeholder="Topic filter (optional)" value={topic} />
        </div>
        <button className="self-end rounded-xl bg-ember px-5 py-3 text-xs font-semibold text-deep-ink transition hover:brightness-110 disabled:cursor-wait disabled:opacity-60" disabled={busy} type="submit">{busy ? "Thinking..." : "Ask tutor →"}</button>
      </form>
      {error && <p aria-live="assertive" className="mt-3 text-right text-xs text-rose-300">{error}</p>}
    </>
  );
}
