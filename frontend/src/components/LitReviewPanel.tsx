import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { BookOpen, CheckCircle2, ChevronDown, Circle, Pill, TestTube } from 'lucide-react'
import { LitReviewResult } from '../api'

interface Props { result: LitReviewResult }

const COLLAPSE_AT = 5

const methodLabels: Record<string, string> = {
  lbdd:   'Ligand-Based (LBDD)',
  sbdd:   'Structure-Based (SBDD)',
  hybrid: 'Hybrid LBDD + SBDD',
}

const methodColors: Record<string, string> = {
  lbdd:   'bg-blue-900/40 text-blue-300 border-blue-800',
  sbdd:   'bg-green-900/40 text-green-300 border-green-800',
  hybrid: 'bg-purple-900/40 text-purple-300 border-purple-800',
}

function statusColor(s: string) {
  if (s === 'FDA Approved' || s === 'EMA Approved' || s === 'Approved') return 'bg-green-900/40 text-green-300 border-green-800'
  if (s.startsWith('Phase III')) return 'bg-teal-900/40 text-teal-300 border-teal-800'
  if (s.startsWith('Phase II'))  return 'bg-yellow-900/40 text-yellow-300 border-yellow-800'
  if (s.startsWith('Phase I'))   return 'bg-orange-900/40 text-orange-300 border-orange-800'
  return 'bg-gray-800/60 text-gray-400 border-gray-700'
}

function ShowMoreList<T>({
  items,
  empty,
  renderItem,
}: {
  items: T[]
  empty: React.ReactNode
  renderItem: (item: T, index: number) => React.ReactNode
}) {
  const [expanded, setExpanded] = useState(false)
  const visible = expanded ? items : items.slice(0, COLLAPSE_AT)
  const hidden = items.length - COLLAPSE_AT

  if (items.length === 0) return <>{empty}</>

  return (
    <div>
      <div className="space-y-2">
        {visible.map((item, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
          >
            {renderItem(item, i)}
          </motion.div>
        ))}
      </div>
      {items.length > COLLAPSE_AT && (
        <button
          onClick={() => setExpanded(e => !e)}
          className="mt-3 flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
        >
          <ChevronDown size={13} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
          {expanded ? 'Show less' : `Show ${hidden} more`}
        </button>
      )}
    </div>
  )
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
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Pill size={16} className="text-janus-400" />
            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Existing Therapies</h4>
          </div>
          {result.existing_therapies.length > 0 && (
            <span className="text-xs text-gray-600">{result.existing_therapies.length} found</span>
          )}
        </div>
        <ShowMoreList
          items={result.existing_therapies}
          empty={
            <p className="text-gray-600 text-sm italic">
              No clinical compounds found in ChEMBL for this target.
            </p>
          }
          renderItem={(t, i) => {
            const sc = statusColor(t.approval_status)
            return (
              <div className="flex items-start justify-between gap-4 p-3 rounded-xl bg-gray-800/40 border border-gray-800">
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
          }}
        />
      </div>

      {/* Key papers */}
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <BookOpen size={16} className="text-janus-400" />
            <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">Key Literature</h4>
          </div>
          {result.discovery_history.key_papers.length > 0 && (
            <span className="text-xs text-gray-600">{result.discovery_history.key_papers.length} papers</span>
          )}
        </div>
        <ShowMoreList
          items={result.discovery_history.key_papers}
          empty={
            <p className="text-gray-600 text-sm italic">
              No publications found in Europe PMC for this target.
            </p>
          }
          renderItem={(p, i) => (
            <li className="flex items-start gap-2.5 text-sm text-gray-400 list-none">
              <span className="text-gray-600 mt-0.5 flex-shrink-0 w-5 text-right">{i + 1}.</span>
              <span>{p}</span>
            </li>
          )}
        />
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
