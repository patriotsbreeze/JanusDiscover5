import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FlaskConical, BookOpen, CheckCircle2, Dna, Atom, Activity,
  ScrollText, AlertCircle, ChevronLeft, Settings
} from 'lucide-react'
import {
  startLitReview, submitApproval, submitMD,
  ApprovalRequest, MDRequest, LLMProvider,
} from '../api'
import { LLMProviderSelector } from '../components/LLMProviderSelector'
import { usePolling } from '../hooks/usePolling'
import { ProgressBar } from '../components/ProgressBar'
import { LitReviewPanel } from '../components/LitReviewPanel'
import { ApprovalPanel } from '../components/ApprovalPanel'
import { ResultsPanel } from '../components/ResultsPanel'
import { MDSetupPanel, MDResultsPanel } from '../components/MDPanel'
import { ManuscriptPanel } from '../components/ManuscriptPanel'

type Step =
  | 'input'
  | 'lit_review'
  | 'approval'
  | 'discovery'
  | 'md_setup'
  | 'md_running'
  | 'done'
  | 'error'

const STEPS = [
  { id: 'input',     icon: Settings,      label: 'Configure' },
  { id: 'approval',  icon: BookOpen,       label: 'Lit Review' },
  { id: 'discovery', icon: Dna,            label: 'Discovery' },
  { id: 'md_setup',  icon: Activity,       label: 'MD Sim' },
  { id: 'done',      icon: ScrollText,     label: 'Manuscript' },
]

export function DiscoveryPage() {
  const [step, setStep] = useState<Step>('input')
  const [protein, setProtein] = useState('')
  const [runMode, setRunMode] = useState<'mock' | 'real'>('mock')
  const [llmProvider, setLlmProvider] = useState<LLMProvider>('anthropic')
  const [llmApiKey, setLlmApiKey] = useState('')
  const [llmModel, setLlmModel] = useState('')
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  const { session, resume } = usePolling(sessionId)

  const handleStart = async () => {
    if (!protein.trim()) return
    setLoading(true)
    setLocalError(null)
    try {
      const { data } = await startLitReview({
        protein_name: protein.trim(),
        run_mode: runMode,
        llm_provider: llmProvider,
        llm_api_key: llmApiKey || undefined,
        llm_model: llmModel || undefined,
      })
      setSessionId(data.session_id)
      setStep('lit_review')
    } catch (e: any) {
      setLocalError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleApprove = async (req: Omit<ApprovalRequest, 'session_id'>) => {
    if (!sessionId) return
    setStep('discovery')
    resume()
    await submitApproval({ ...req, session_id: sessionId })
  }

  const handleMD = async (req: Omit<MDRequest, 'session_id'>) => {
    if (!sessionId) return
    setStep(req.run_md ? 'md_running' : 'done')
    resume()
    await submitMD({ ...req, session_id: sessionId })
  }

  // Sync step with server status
  const status = session?.status
  if (status === 'awaiting_approval' && step === 'lit_review') setStep('approval')
  if (status === 'awaiting_md_decision' && step === 'discovery') setStep('md_setup')
  if (status === 'manuscript_done' && (step === 'md_running' || step === 'md_setup')) setStep('done')
  if (status === 'error' && step !== 'error') setStep('error')

  const isRunning = status && ['lit_review_running', 'discovery_running', 'md_running'].includes(status)

  const activeStepIdx = STEPS.findIndex(s => {
    if (step === 'input') return s.id === 'input'
    if (step === 'lit_review' || step === 'approval') return s.id === 'approval'
    if (step === 'discovery') return s.id === 'discovery'
    if (step === 'md_setup' || step === 'md_running') return s.id === 'md_setup'
    if (step === 'done') return s.id === 'done'
    return false
  })

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* Top nav */}
      <nav className="border-b border-gray-800 px-6 py-3 flex items-center justify-between sticky top-0 z-20 bg-gray-950/95 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-janus-500 to-bio-green flex items-center justify-center">
            <FlaskConical size={14} className="text-white" />
          </div>
          <span className="font-bold text-white">JanusDiscover</span>
        </div>

        {/* Progress stepper */}
        <div className="hidden md:flex items-center gap-1">
          {STEPS.map((s, i) => {
            const done = i < activeStepIdx
            const active = i === activeStepIdx
            return (
              <div key={s.id} className="flex items-center gap-1">
                <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
                  active  ? 'bg-janus-700 text-white' :
                  done    ? 'bg-gray-800 text-bio-green' :
                  'bg-gray-900 text-gray-600'
                }`}>
                  {done ? <CheckCircle2 size={12} /> : <s.icon size={12} />}
                  {s.label}
                </div>
                {i < STEPS.length - 1 && (
                  <div className={`w-6 h-px ${i < activeStepIdx ? 'bg-gray-600' : 'bg-gray-800'}`} />
                )}
              </div>
            )
          })}
        </div>

        <div className="flex items-center gap-2 text-sm">
          {runMode === 'mock'
            ? <span className="badge bg-yellow-900/40 text-yellow-300 border border-yellow-800">MOCK</span>
            : <span className="badge bg-green-900/40 text-green-300 border border-green-800">REAL</span>
          }
          {sessionId && (
            <span className="text-gray-600 font-mono text-xs">{sessionId.slice(0, 8)}</span>
          )}
        </div>
      </nav>

      {/* Main content */}
      <div className="flex-1 max-w-3xl mx-auto w-full px-4 py-8">
        <AnimatePresence mode="wait">

          {/* ── STEP 0: Input ─────────────────────────────────────────────────── */}
          {step === 'input' && (
            <motion.div key="input" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -16 }}>
              <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-2">Start Drug Discovery</h1>
                <p className="text-gray-400">Enter a protein target to begin the JanusDiscover pipeline.</p>
              </div>

              <div className="card space-y-5">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Protein / Target Name</label>
                  <input
                    className="input-field text-lg"
                    placeholder="e.g. ABL1 kinase, EGFR, CDK2, BRAF…"
                    value={protein}
                    onChange={e => setProtein(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleStart()}
                  />
                  <p className="text-gray-600 text-xs mt-1.5">
                    JanusDiscover will query ChEMBL, RCSB PDB, and use AI to synthesise a literature review.
                  </p>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">Run Mode</label>
                  <div className="grid grid-cols-2 gap-3">
                    {(['mock', 'real'] as const).map(m => (
                      <button
                        key={m}
                        onClick={() => setRunMode(m)}
                        className={`p-4 rounded-xl border text-left transition-all ${
                          runMode === m ? 'border-janus-600 bg-janus-950/50' : 'border-gray-800 hover:border-gray-700'
                        }`}
                      >
                        <div className="flex items-center gap-2 mb-1">
                          <div className={`w-3 h-3 rounded-full ${m === 'mock' ? 'bg-yellow-400' : 'bg-bio-green'}`} />
                          <span className="font-semibold text-white uppercase text-sm">{m}</span>
                        </div>
                        <p className="text-gray-500 text-xs">
                          {m === 'mock'
                            ? 'Full workflow demo with realistic pre-computed results'
                            : 'Live computation: ChEMBL, AutoDock Vina, OpenMM'}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>

                {runMode === 'real' && (
                  <LLMProviderSelector
                    disabled={loading}
                    onChange={(p, k, m) => { setLlmProvider(p); setLlmApiKey(k); setLlmModel(m) }}
                  />
                )}

                {localError && (
                  <div className="flex items-center gap-2 p-3 rounded-xl bg-red-950/30 border border-red-900 text-red-300 text-sm">
                    <AlertCircle size={16} className="flex-shrink-0" />
                    {localError}
                  </div>
                )}

                <button
                  className="btn-primary w-full text-base py-3"
                  onClick={handleStart}
                  disabled={loading || !protein.trim()}
                >
                  {loading ? '⏳ Starting…' : '🔬 Begin Discovery Pipeline'}
                </button>
              </div>

              {/* Example targets */}
              <div className="mt-6">
                <p className="text-gray-600 text-sm mb-2">Try an example target:</p>
                <div className="flex flex-wrap gap-2">
                  {['ABL1 kinase', 'EGFR', 'CDK2', 'BRAF V600E', 'SARS-CoV-2 Mpro', 'HIV protease'].map(t => (
                    <button
                      key={t}
                      onClick={() => setProtein(t)}
                      className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-400 text-xs border border-gray-700 transition-colors"
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── STEP 1: Lit Review running ─────────────────────────────────────── */}
          {step === 'lit_review' && (
            <motion.div key="lit_running" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SectionHeader
                icon={BookOpen}
                title="Literature Review"
                subtitle={`Analysing ${protein} — querying ChEMBL, RCSB PDB, and AI synthesis`}
              />
              <div className="card mt-4">
                <ProgressBar
                  pct={session?.progress_pct ?? 10}
                  message={session?.progress_msg ?? 'Running literature review…'}
                />
                <p className="text-gray-600 text-xs mt-3 text-center">
                  This may take 10–30 seconds in real mode…
                </p>
              </div>
            </motion.div>
          )}

          {/* ── STEP 2: Approval ──────────────────────────────────────────────── */}
          {step === 'approval' && session?.lit_review && (
            <motion.div key="approval" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SectionHeader
                icon={BookOpen}
                title="Literature Review Complete"
                subtitle="Review the findings below and approve the discovery run"
              />
              <div className="mt-4 space-y-4">
                <LitReviewPanel result={session.lit_review} />
                <ApprovalPanel
                  sessionId={sessionId!}
                  litReview={session.lit_review}
                  runMode={runMode}
                  onApprove={handleApprove}
                  onCancel={() => setStep('input')}
                />
              </div>
            </motion.div>
          )}

          {/* ── STEP 3: Discovery running ──────────────────────────────────────── */}
          {step === 'discovery' && (
            <motion.div key="discovery" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SectionHeader
                icon={Dna}
                title="Running Discovery Pipeline"
                subtitle={`Executing ${runMode === 'mock' ? 'mock' : 'real'} LBDD/SBDD virtual screening`}
              />
              <div className="card mt-4 space-y-4">
                <ProgressBar
                  pct={session?.progress_pct ?? 10}
                  message={session?.progress_msg ?? 'Screening library…'}
                />
                <div className="grid grid-cols-3 gap-3 text-center">
                  {['Fingerprint generation', 'ML training (5-fold CV)', 'Library screening'].map(s => (
                    <div key={s} className="p-3 rounded-xl bg-gray-800/40 border border-gray-800">
                      <div className="w-4 h-4 rounded-full bg-janus-600 animate-pulse-slow mx-auto mb-2" />
                      <p className="text-gray-500 text-xs">{s}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── STEP 4: MD Setup ──────────────────────────────────────────────── */}
          {step === 'md_setup' && session?.discovery_result && (
            <motion.div key="md_setup" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SectionHeader
                icon={Dna}
                title="Discovery Results"
                subtitle="Virtual screening complete — review results and optionally run MD"
              />
              <div className="mt-4 space-y-4">
                <ResultsPanel result={session.discovery_result} sessionId={sessionId!} />
                {session.manuscript && <ManuscriptPanel result={session.manuscript} sessionId={sessionId!} />}
                <MDSetupPanel
                  sessionId={sessionId!}
                  runMode={runMode}
                  onSubmit={handleMD}
                />
              </div>
            </motion.div>
          )}

          {/* ── STEP 5: MD Running ────────────────────────────────────────────── */}
          {step === 'md_running' && (
            <motion.div key="md_running" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SectionHeader
                icon={Activity}
                title="MD Simulation Running"
                subtitle="OpenMM 8 all-atom simulation in progress…"
              />
              <div className="card mt-4 space-y-4">
                <ProgressBar
                  pct={session?.progress_pct ?? 15}
                  message={session?.progress_msg ?? 'Simulating…'}
                  color="green"
                />
                <div className="grid grid-cols-4 gap-3 text-center">
                  {['Solvation', 'Minimisation', 'Equilibration', 'Production'].map(s => (
                    <div key={s} className="p-3 rounded-xl bg-gray-800/40 border border-gray-800">
                      <div className="w-4 h-4 rounded-full bg-bio-green animate-pulse-slow mx-auto mb-2" />
                      <p className="text-gray-500 text-xs">{s}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* ── STEP 6: Done ──────────────────────────────────────────────────── */}
          {step === 'done' && session && (
            <motion.div key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <SectionHeader
                icon={ScrollText}
                title="Discovery Complete"
                subtitle="All results, figures, and manuscript are ready"
              />
              <div className="mt-4 space-y-4">
                {session.discovery_result && (
                  <ResultsPanel result={session.discovery_result} sessionId={sessionId!} />
                )}
                {session.md_result && (
                  <MDResultsPanel result={session.md_result} sessionId={sessionId!} />
                )}
                {session.manuscript && (
                  <ManuscriptPanel result={session.manuscript} sessionId={sessionId!} />
                )}
                <div className="text-center pt-4">
                  <button
                    className="btn-secondary"
                    onClick={() => { setStep('input'); setSessionId(null) }}
                  >
                    ← Start New Discovery
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {/* ── Error ─────────────────────────────────────────────────────────── */}
          {step === 'error' && (
            <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="card border-red-900 bg-red-950/20">
                <div className="flex items-start gap-3">
                  <AlertCircle size={20} className="text-red-400 flex-shrink-0 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-white mb-1">Pipeline Error</h3>
                    <p className="text-red-300 text-sm">{session?.error_message ?? localError ?? 'An unexpected error occurred.'}</p>
                    <button
                      className="btn-secondary mt-4"
                      onClick={() => { setStep('input'); setSessionId(null) }}
                    >
                      ← Try Again
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </div>
    </div>
  )
}

function SectionHeader({ icon: Icon, title, subtitle }: {
  icon: any; title: string; subtitle: string
}) {
  return (
    <div className="mb-2">
      <div className="flex items-center gap-3 mb-1">
        <div className="w-9 h-9 rounded-xl bg-janus-900/50 border border-janus-800 flex items-center justify-center">
          <Icon size={18} className="text-janus-400" />
        </div>
        <h2 className="text-2xl font-bold text-white">{title}</h2>
      </div>
      <p className="text-gray-400 text-sm ml-12">{subtitle}</p>
    </div>
  )
}
