export const STUDENT_ID = "demo-student";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8050")
  .replace(/\/+$/, "");

export interface Citation {
  answer: string;
  verdict: "answer" | "refuse";
  citations: string[];
}

export interface IngestResult {
  file: string;
  kind: string;
  units: number;
  errors: string[];
}

export interface QuizQuestion {
  qtype: "mcq" | "short" | "numerical";
  stem: string;
  options: string[];
  answer_key: string;
  topic: string;
  source: string;
  difficulty: string;
  explanation: string;
  qhash: string;
}

export interface QuizResult {
  qhash: string;
  correct: boolean;
  method: string;
  given: string;
  key: string;
  source: string;
  verify: string;
  feedback: string;
}

export interface QuizFeedback {
  results: QuizResult[];
  weak_topics: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new Error(
      `Could not reach the backend at ${API_BASE_URL}. Start FastAPI and try again.`,
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Keep the HTTP status message when the response has no JSON body.
    }
    throw new Error(detail);
  }

  return (await response.json()) as T;
}

export function ingestFile(file: File): Promise<IngestResult> {
  const data = new FormData();
  data.append("file", file);
  return request<IngestResult>("/ingest", { method: "POST", body: data });
}

export function askTutor(question: string, topic?: string): Promise<Citation> {
  return request<Citation>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, topic, student_id: STUDENT_ID }),
  });
}

export function generateQuiz(scope: string, n: number): Promise<{ questions: QuizQuestion[] }> {
  return request<{ questions: QuizQuestion[] }>("/quiz", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scope, student_id: STUDENT_ID, difficulty: "auto", n }),
  });
}

export function submitQuiz(
  scope: string,
  questions: QuizQuestion[],
  answers: Record<string, string>,
): Promise<QuizFeedback> {
  return request<QuizFeedback>("/quiz/submit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scope, student_id: STUDENT_ID, questions, answers }),
  });
}

export function getMastery(): Promise<Record<string, number>> {
  return request<Record<string, number>>(`/mastery/${encodeURIComponent(STUDENT_ID)}`);
}
