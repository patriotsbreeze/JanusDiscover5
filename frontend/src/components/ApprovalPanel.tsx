import { useState } from 'react'
import { motion } from 'framer-motion'
import { CheckCircle2, Database, FlaskConical, Layers } from 'lucide-react'
import { LitReviewResult, ApprovalRequest } from '../api'

interface Props {
  sessionId: string
  litReview: LitReviewResult
  runMode: 'mock' | 'real'
  onApprove: (req: Omit<ApprovalRequest, 'session_id'>) => void
  onCancel: () => void
}

const DATASETS = [
  { value: 'zinc_250k', label: 'ZINC-250k', desc: '250,000 drug-like compounds' },
  { value: 'fda_approved', label: 'FDA Approved', desc: '~2,300 approved small molecules' },
  { value: 'chembl', label: 'ChEMBL', desc: 'Bioactive compound database' },
  { value: 'custom', label: 'Custom Upload', desc: 'Provide your own SMILES file' },
] as const

const METHODS = [
  { value: 'lbdd',   label: 'LBDD',          desc: 'Morgan fingerprints + Random Forest ensemble' },
  { value: 'sbdd',   label: 'SBDD',           desc: 'AutoDock Vina 1.2 molecular docking' },
  { value: 'hybrid', label: 'Hybrid',         desc: 'Consensus LBDD + SBDD scoring (recommended)' },
] as const

export function ApprovalPanel({ sessionId, litReview, runMode, onApprove, onCancel }: Props) {
  const [method, setMethod] = useState<'lbdd' | 'sbdd' | 'hybrid'>(litReview.recommended_method)
  const [dataset, setDataset] = useState<ApprovalRequest['dataset']>('zinc_250k')

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      <div className="card border-janus-700 glow-blue">
        <h3 className="font-semibold text-white text-lg mb-1 flex items-center gap-2">
          <CheckCircle2 size={20} className="text-janus-400" />
          Approve Discovery Run
        </h3>
        <p className="text-gray-400 text-sm">
          Review the literature analysis above, then configure and approve the discovery pipeline.
        </p>
      </div>

      {/* Method selector */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3 flex items-center gap-2">
          <FlaskConical size={14} className="text-janus-400" />
          Discovery Method
        </h4>
        <div className="space-y-2">
          {METHODS.map(m => (
            <button
              key={m.value}
              onClick={() => setMethod(m.value)}
              className={`w-full flex items-start gap-3 p-3 rounded-xl border text-left transition-all ${
                method === m.value
                  ? 'border-janus-600 bg-janus-950/50'
                  : 'border-gray-800 bg-gray-800/20 hover:border-gray-700'
              }`}
            >
              <div className={`w-4 h-4 rounded-full border-2 mt-0.5 flex-shrink-0 ${
                method === m.value ? 'border-janus-400 bg-janus-500' : 'border-gray-600'
              }`} />
              <div>
                <span className="font-medium text-white text-sm">{m.label}</span>
                {m.value === litReview.recommended_method && (
                  <span className="ml-2 badge bg-janus-900/50 text-janus-300 border border-janus-800 text-xs">AI Recommended</span>
                )}
                <p className="text-gray-500 text-xs mt-0.5">{m.desc}</p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Dataset selector */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3 flex items-center gap-2">
          <Database size={14} className="text-janus-400" />
          Screening Dataset
        </h4>
        <div className="grid grid-cols-2 gap-2">
          {DATASETS.map(d => (
            <button
              key={d.value}
              onClick={() => setDataset(d.value)}
              className={`flex flex-col items-start p-3 rounded-xl border text-left transition-all ${
                dataset === d.value
                  ? 'border-janus-600 bg-janus-950/50'
                  : 'border-gray-800 bg-gray-800/20 hover:border-gray-700'
              }`}
            >
              <span className="font-medium text-white text-sm">{d.label}</span>
              <span className="text-gray-500 text-xs mt-0.5">{d.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Run mode info */}
      <div className={`card border ${runMode === 'mock' ? 'border-yellow-800 bg-yellow-950/20' : 'border-green-800 bg-green-950/20'}`}>
        <div className="flex items-center gap-3">
          <span className={`badge border ${runMode === 'mock' ? 'bg-yellow-900/40 text-yellow-300 border-yellow-800' : 'bg-green-900/40 text-green-300 border-green-800'}`}>
            {runMode.toUpperCase()} MODE
          </span>
          <p className="text-gray-400 text-sm">
            {runMode === 'mock'
              ? 'Simulated workflow with realistic pre-computed results. No external API calls.'
              : 'Live pipeline: ChEMBL queries, AutoDock Vina docking, real computation.'}
          </p>
        </div>
      </div>

      {/* Action buttons */}
      <div className="flex gap-3">
        <button
          className="btn-primary flex-1"
          onClick={() => onApprove({ approved: true, chosen_method: method, dataset, run_mode: runMode })}
        >
          ✓ Approve & Run {method.toUpperCase()} Discovery
        </button>
        <button className="btn-danger" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </motion.div>
  )
}
