import { motion } from 'framer-motion'
import { BookOpen, CheckCircle2, Circle, ExternalLink, Pill, TestTube } from 'lucide-react'
import { LitReviewResult } from '../api'

interface Props { result: LitReviewResult }

const methodLabels: Record<string, string> = {
  lbdd: 'Ligand-Based (LBDD)',
  sbdd: 'Structure-Based (SBDD)',
  hybrid: 'Hybrid LBDD + SBDD',
}

const methodColors: Record<string, string> = {
  lbdd:   'bg-blue-900/40 text-blue-300 border-blue-800',
  sbdd:   'bg-green-900/40 text-green-300 border-green-800',
  hybrid: 'bg-purple-900/40 text-purple-300 border-purple-800',
}

const statusColors: Record<string, string> = {
  'FDA Approved':   'bg-green-900/40 text-green-300 border-green-800',
  'Clinical Trial': 'bg-yellow-900/40 text-yellow-300 border-yellow-800',
  'Preclinical':    'bg-orange-900/40 text-orange-300 border-orange-800',
}

export function LitReviewPanel({ result }: Props) {
  const mc = methodColors[result.recommended_method] ?? methodColors.hybrid

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="space-y-5"
    >
      {/* Target summary */}
      <div className="card glow-blue">
        <div className="flex items-start gap-4">
          <div className="w-10 h-10 rounded-xl bg-janus-900/50 border border-janus-800 flex items-center justify-center flex-shrink-0">
            <TestTube size={20} className="text-janus-400" />
          </div>
          <div>
            <h3 className="font-semibold text-white text-lg mb-1">{result.protein_name}</h3>
            <p className="text-gray-400 text-sm leading-relaxed">{result.description}</p>
            <div className="flex flex-wrap gap-2 mt-3">
              <span className="badge bg-gray-800 text-gray-300 border border-gray-700">
                {result.known_actives_count} known actives (ChEMBL)
              </span>
              {result.available_pdb_ids.map(id => (
                <span key={id} className="badge bg-gray-800 text-gray-300 border border-gray-700">
                  PDB: {id}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Prior work flags */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">
          Prior Computational Work
        </h4>
        <div className="grid grid-cols-3 gap-3">
          {([
            ['LBDD', result.discovery_history.lbdd_done],
            ['SBDD', result.discovery_history.sbdd_done],
            ['Hybrid', result.discovery_history.hybrid_done],
          ] as [string, boolean][]).map(([label, done]) => (
            <div key={label} className={`flex items-center gap-2 p-3 rounded-xl border ${done ? 'bg-green-950/30 border-green-900' : 'bg-gray-800/30 border-gray-800'}`}>
              {done
                ? <CheckCircle2 size={16} className="text-bio-green flex-shrink-0" />
                : <Circle size={16} className="text-gray-600 flex-shrink-0" />}
              <span className={`text-sm font-medium ${done ? 'text-green-300' : 'text-gray-500'}`}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Existing therapies */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Pill size={16} className="text-janus-400" />
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
            Existing Therapies
          </h4>
        </div>
        <div className="space-y-2">
          {result.existing_therapies.map((t, i) => {
            const sc = statusColors[t.approval_status] ?? statusColors['Preclinical']
            return (
              <div key={i} className="flex items-start justify-between gap-4 p-3 rounded-xl bg-gray-800/40 border border-gray-800">
                <div>
                  <span className="font-medium text-white text-sm">{t.name}</span>
                  <p className="text-gray-500 text-xs mt-0.5">{t.mechanism}</p>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  {t.year && <span className="text-gray-600 text-xs">{t.year}</span>}
                  <span className={`badge border text-xs ${sc}`}>{t.approval_status}</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Key papers */}
      <div className="card">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen size={16} className="text-janus-400" />
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Key Literature</h4>
        </div>
        <ul className="space-y-1.5">
          {result.discovery_history.key_papers.map((p, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-gray-400">
              <span className="text-gray-600 mt-0.5 flex-shrink-0">{i + 1}.</span>
              {p}
            </li>
          ))}
        </ul>
      </div>

      {/* Recommendation */}
      <div className={`card border-2 ${mc.replace('bg-', 'border-').split(' ')[0]}`}>
        <div className="flex items-start gap-4">
          <span className={`badge border text-sm ${mc} flex-shrink-0 mt-0.5`}>
            Recommended
          </span>
          <div>
            <p className="font-semibold text-white mb-1">
              {methodLabels[result.recommended_method]}
            </p>
            <p className="text-gray-400 text-sm leading-relaxed">
              {result.recommendation_rationale}
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
