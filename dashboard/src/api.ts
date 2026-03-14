import { API_BASE_URL } from "./App";

export interface Experiment {
  experiment_id: string;
  name: string;
  status: string;
  config: Record<string, unknown>;
  best_value?: number | null;
  best_params?: Record<string, unknown> | null;
  best_model_path?: string | null;
}

export interface MetricRecord {
  metric_name: string;
  value: number;
  step: number;
  timestamp: number;
}

export interface StudySummaryResponse {
  study_name: string;
  best_value: number | null;
  best_params: Record<string, unknown> | null;
  best_model_path?: string | null;
  n_trials: number;
  summary: Record<string, unknown>;
  history: [number, number][];
}

export interface StudyImportanceResponse {
  study_name: string;
  importance: Record<string, number>;
}

export interface TrialDetail {
  trial_id: number;
  params: Record<string, unknown>;
  value: number | null;
  state: string;
  duration: number | null;
  error: string | null;
}

export interface StudyTrialsResponse {
  study_name: string;
  direction: string;
  n_completed: number;
  n_failed: number;
  n_trials: number;
  best_value: number | null;
  best_model_path: string | null;
  trials: TrialDetail[];
}

async function request<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function fetchExperiments(status?: string): Promise<Experiment[]> {
  const q = status ? `?status=${encodeURIComponent(status)}` : "";
  return request<Experiment[]>(`/experiments${q}`);
}

export async function fetchExperiment(id: string): Promise<Experiment> {
  return request<Experiment>(`/experiments/${encodeURIComponent(id)}`);
}

export async function fetchMetrics(experimentId: string, name?: string) {
  const q = name ? `?name=${encodeURIComponent(name)}` : "";
  return request<{ experiment_id: string; metrics: MetricRecord[] }>(
    `/experiments/${encodeURIComponent(experimentId)}/metrics${q}`
  );
}

export async function fetchStudy(studyName: string): Promise<StudySummaryResponse> {
  return request<StudySummaryResponse>(`/studies/${encodeURIComponent(studyName)}`);
}

export async function fetchStudyImportance(
  studyName: string
): Promise<StudyImportanceResponse> {
  return request<StudyImportanceResponse>(`/studies/${encodeURIComponent(studyName)}/importance`);
}

export async function fetchStudyTrials(
  studyName: string
): Promise<StudyTrialsResponse> {
  return request<StudyTrialsResponse>(`/studies/${encodeURIComponent(studyName)}/trials`);
}

