import axios from 'axios'

const BASE = '/api'

export const api = axios.create({ baseURL: BASE })

export type LLMProvider = 'anthropic' | 'openai' | 'google' | 'deepseek' | 'ollama'

export interface LitReviewRequest {
  protein_name: string
  run_mode: 'mock' | 'real'
  llm_provider?: LLMProvider
  llm_api_key?: string
  llm_model?: string
}

export interface ApprovalRequest {
  session_id: string
  approved: boolean
  chosen_method?: 'lbdd' | 'sbdd' | 'hybrid'
  dataset: 'fda_approved' | 'zinc_250k' | 'chembl' | 'custom'
  run_mode: 'mock' | 'real'
}

export interface MDRequest {
  session_id: string
  run_md: boolean
  duration?: { value: number; unit: 'ps' | 'ns' }
  run_mode: 'mock' | 'real'
}

export interface SessionState {
  session_id: string
  protein_name: string
  run_mode: string
  status: string
  progress_pct: number
  progress_msg: string
  lit_review: LitReviewResult | null
  discovery_result: DiscoveryResult | null
  md_result: MDResult | null
  manuscript: ManuscriptResult | null
  error_message: string | null
}

export interface LitReviewResult {
  session_id: string
  protein_name: string
  description: string
  existing_therapies: TherapyEntry[]
  discovery_history: DiscoveryHistory
  recommended_method: 'lbdd' | 'sbdd' | 'hybrid'
  recommendation_rationale: string
  available_pdb_ids: string[]
  known_actives_count: number
}

export interface TherapyEntry {
  name: string
  mechanism: string
  approval_status: string
  year?: number
}

export interface DiscoveryHistory {
  lbdd_done: boolean
  sbdd_done: boolean
  hybrid_done: boolean
  key_papers: string[]
  existing_scaffolds: string[]
}

export interface CompoundResult {
  compound_id: string
  smiles: string
  score: number
  predicted_activity?: number
  docking_score?: number
  binding_affinity_kcal?: number
}

export interface ValidationMetrics {
  auc_roc: number
  bedroc: number
  ef1_percent: number
  ef5_percent: number
  enrichment_auc?: number
  pose_rmsd_mean?: number
  pose_success_rate_2A?: number
}

export interface DiscoveryResult {
  session_id: string
  method_used: string
  dataset_used: string
  compounds_screened: number
  top_hits: CompoundResult[]
  validation_metrics: ValidationMetrics
  figures: string[]
  run_time_seconds: number
  mock: boolean
}

export interface MDResult {
  duration_ns: number
  frames_analyzed: number
  rmsd_mean_nm: number
  rmsd_std_nm: number
  rmsf_mean_nm: number
  radius_of_gyration_mean_nm: number
  potential_energy_mean_kj_mol: number
  figures: string[]
  mock: boolean
}

export interface ManuscriptResult {
  session_id: string
  tex_path: string
  pdf_path: string | null
  bib_path: string
}

export const startLitReview = (req: LitReviewRequest) =>
  api.post<{ session_id: string; status: string }>('/lit-review', req)

export const getSession = (sessionId: string) =>
  api.get<SessionState>(`/session/${sessionId}`)

export const submitApproval = (req: ApprovalRequest) =>
  api.post('/approve', req)

export const submitMD = (req: MDRequest) =>
  api.post('/md', req)

export const figureUrl = (sessionId: string, filename: string) =>
  `${BASE}/figures/${sessionId}/${filename}`

export const manuscriptUrl = (sessionId: string, type: 'tex' | 'pdf' | 'bib') =>
  `${BASE}/manuscript/${sessionId}/${type}`
