import { useCallback, useEffect, useState } from "react";
import { getMastery, STUDENT_ID } from "../../lib/api";

const currentState = {
  student_id: STUDENT_ID,
  model: "Bayesian Knowledge Tracing",
};

export function PageMastery() {
  const [mastery, setMastery] = useState<Record<string, number>>({});
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      setMastery(await getMastery());
      setError("");
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load mastery.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    window.addEventListener("learning-companion:mastery-updated", refresh);
    return () => window.removeEventListener("learning-companion:mastery-updated", refresh);
  }, [refresh]);

  return (
    <>
      <p className="mt-3 text-sm text-warm-cream/60">Your quiz performance updates mastery per topic.</p>
      {error && <p aria-live="assertive" className="mt-4 text-sm text-rose-300">{error}</p>}
      <div aria-live="polite" className="mt-7 rounded-xl border border-white/10 bg-black/20 p-4">
        {loading ? <p className="text-xs text-warm-cream/50">Loading mastery...</p> :
          Object.keys(mastery).length ? (
            <ul className="space-y-3">
              {Object.entries(mastery).map(([topic, score]) => (
                <li className="flex items-center gap-3 text-sm" key={topic}>
                  <span className="min-w-0 flex-1 text-warm-cream/80">{topic}</span>
                  <progress className="h-2 w-28 accent-[#E38343]" max={1} value={score} />
                  <span className="w-10 text-right font-mono text-xs text-ember">{Math.round(score * 100)}%</span>
                </li>
              ))}
            </ul>
          ) : <p className="text-center text-xs text-warm-cream/40">No mastery data yet. Complete a quiz to start tracking.</p>}
      </div>
      <div className="mt-8">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="font-display text-xl text-warm-cream">Notebook state</h2>
          <span className="font-mono text-[9px] uppercase tracking-widest text-warm-cream/35">Current state</span>
        </div>
        <pre className="overflow-x-auto rounded-xl border border-white/10 bg-black/35 p-4 font-mono text-[10px] leading-5 text-[#e5b487] sm:text-xs"><code>{JSON.stringify({ ...currentState, topics: mastery }, null, 2)}</code></pre>
      </div>
    </>
  );
}
