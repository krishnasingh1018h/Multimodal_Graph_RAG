import { useRef, useState, type FormEvent } from "react";
import { askTutor, ingestFile, type Citation } from "../../lib/api";

export function HeroIngestionCard() {
  const fileInput = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState<Citation | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim() && files.length === 0) return;
    setBusy(true);
    setMessage("");
    setAnswer(null);
    try {
      for (const file of files) {
        const result = await ingestFile(file);
        setFiles((current) => current.filter((selected) => selected !== file));
        setMessage((current) =>
          `${current}${current ? " " : ""}Indexed ${result.file}: ${result.units} units.`,
        );
      }
      if (query.trim()) {
        setAnswer(await askTutor(query.trim()));
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The request failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="mt-6 w-full max-w-2xl rounded-2xl border border-white/10 bg-black/40 p-3 shadow-[0_24px_90px_rgba(0,0,0,0.38)] backdrop-blur-xl sm:mt-8 sm:p-4"
      onSubmit={submit}
    >
      <label className="sr-only" htmlFor="hero-query">Ask how your concepts are connected</label>
      <textarea
        className="min-h-14 w-full resize-y bg-transparent px-3 py-2 text-sm leading-6 text-warm-cream outline-none placeholder:text-warm-cream/50 focus-visible:ring-1 focus-visible:ring-ember/60 disabled:opacity-60"
        id="hero-query"
        onChange={(event) => setQuery(event.target.value)}
        placeholder="How are these concepts connected?"
        rows={2}
        value={query}
      />
      {files.length > 0 && (
        <ul aria-label="Selected local files" className="flex flex-wrap gap-2 px-3 pb-3">
          {files.map((file) => (
            <li className="rounded-full border border-ember/25 bg-ember/[0.08] px-3 py-1 font-mono text-[10px] text-warm-cream/75" key={`${file.name}-${file.size}`}>
              {file.name}
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 px-2 pt-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            accept=".pdf,.ppt,.pptx,.txt,.srt,.vtt,.jpg,.jpeg,.png"
            className="sr-only"
            multiple
            onChange={(event) => {
              const selected = Array.from(event.currentTarget.files ?? []);
              setFiles((current) => [...current, ...selected]);
              event.currentTarget.value = "";
            }}
            ref={fileInput}
            type="file"
          />
          <button
            className="rounded-full border border-white/15 px-3 py-2 text-[11px] text-warm-cream/80 transition hover:border-ember/60 hover:text-warm-cream"
            onClick={() => fileInput.current?.click()}
            type="button"
          >
            <span aria-hidden="true" className="mr-1.5 text-ember">📎</span> Upload material
          </button>
          <span className="hidden font-mono text-[9px] uppercase tracking-[0.12em] text-warm-cream/40 sm:inline">PDF · slides · notes · images</span>
        </div>
        <button className="inline-flex items-center gap-2 rounded-full bg-ember px-4 py-2.5 text-[11px] font-semibold text-deep-ink transition-opacity hover:opacity-90 disabled:cursor-wait disabled:opacity-60 sm:px-5" disabled={busy} type="submit">
          {busy ? "Working..." : "Explore connections →"}
        </button>
      </div>
      {message && <p aria-live="polite" className="px-3 pt-3 text-xs text-warm-cream/55">{message}</p>}
      {answer && (
        <div aria-live="polite" className="mt-4 rounded-xl border border-ember/20 bg-black/30 p-4">
          <p className="font-mono text-[9px] uppercase tracking-wider text-ember">
            {answer.verdict === "answer" ? "Answer" : "Not covered by your material"}
          </p>
          <p className="mt-2 whitespace-pre-wrap text-sm leading-6 text-warm-cream/85">{answer.answer}</p>
          {answer.citations.length > 0 && (
            <p className="mt-3 text-xs text-warm-cream/55">Sources: {answer.citations.join("; ")}</p>
          )}
        </div>
      )}
    </form>
  );
}
