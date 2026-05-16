import { motion } from 'framer-motion'
import { BarChart3, Download, Layers, TrendingUp, Zap } from 'lucide-react'
import { DiscoveryResult, figureUrl } from '../api'

interface Props {
  result: DiscoveryResult
  sessionId: string
}

const methodColor: Record<string, string> = {
  lbdd:   'bg-blue-900/40 text-blue-300 border-blue-800',
  sbdd:   'bg-green-900/40 text-green-300 border-green-800',
  hybrid: 'bg-purple-900/40 text-purple-300 border-purple-800',
}

export function ResultsPanel({ result, sessionId }: Props) {
  const mc = methodColor[result.method_used] ?? methodColor.hybrid
  const vm = result.validation_metrics

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      {/* Summary header */}
      <div className="card glow-green border-green-900">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`badge border ${mc}`}>{result.method_used.toUpperCase()}</span>
              {result.mock && <span className="badge bg-yellow-900/40 text-yellow-300 border-yellow-800">MOCK</span>}
            </div>
            <h3 className="font-semibold text-white text-lg">Discovery Complete</h3>
            <p className="text-gray-400 text-sm">
              Screened {result.compounds_screened.toLocaleString()} compounds in {result.run_time_seconds.toFixed(1)}s
            </p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-white">{result.top_hits.length}</div>
            <div className="text-gray-500 text-sm">top hits</div>
          </div>
        </div>
      </div>

      {/* Validation metrics */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4 flex items-center gap-2">
          <TrendingUp size={14} className="text-janus-400" />
          Validation Metrics
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <MetricCard label="AUC-ROC" value={vm.auc_roc.toFixed(3)} sub="≥0.7 good" color="blue" />
          <MetricCard label="BEDROC" value={vm.bedroc.toFixed(3)} sub="α=20" color="purple" />
          <MetricCard label="EF at 1%" value={vm.ef1_percent.toFixed(2) + '×'} sub="enrichment" color="green" />
          <MetricCard label="EF at 5%" value={vm.ef5_percent.toFixed(2) + '×'} sub="enrichment" color="green" />
          {vm.pose_rmsd_mean != null && (
            <MetricCard label="Pose RMSD" value={vm.pose_rmsd_mean.toFixed(2) + ' Å'} sub="mean redocking" color="orange" />
          )}
          {vm.pose_success_rate_2A != null && (
            <MetricCard label="Success Rate" value={(vm.pose_success_rate_2A * 100).toFixed(0) + '%'} sub="RMSD < 2 Å" color="orange" />
          )}
        </div>
      </div>

      {/* Top hits table */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4 flex items-center gap-2">
          <Zap size={14} className="text-janus-400" />
          Top-20 Virtual Screening Hits
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-500 border-b border-gray-800">
                <th className="text-left pb-2 font-medium">Rank</th>
                <th className="text-left pb-2 font-medium">ID</th>
                <th className="text-left pb-2 font-medium">Score</th>
                {result.top_hits[0]?.docking_score != null && (
                  <th className="text-left pb-2 font-medium">ΔG (kcal/mol)</th>
                )}
                {result.top_hits[0]?.predicted_activity != null && (
                  <th className="text-left pb-2 font-medium">pActivity</th>
                )}
                <th className="text-left pb-2 font-medium">SMILES</th>
              </tr>
            </thead>
            <tbody>
              {result.top_hits.slice(0, 20).map((hit, i) => (
                <tr key={hit.compound_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-2 text-gray-500 font-mono">#{i + 1}</td>
                  <td className="py-2 text-janus-400 font-mono text-xs">{hit.compound_id}</td>
                  <td className="py-2 text-white font-semibold">{hit.score.toFixed(3)}</td>
                  {hit.docking_score != null && (
                    <td className="py-2 text-bio-green">{hit.docking_score.toFixed(1)}</td>
                  )}
                  {hit.predicted_activity != null && (
                    <td className="py-2 text-blue-300">{hit.predicted_activity.toFixed(2)}</td>
                  )}
                  <td className="py-2 text-gray-500 font-mono text-xs max-w-xs truncate"
                      title={hit.smiles}>
                    {hit.smiles.slice(0, 40)}{hit.smiles.length > 40 ? '…' : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Figures */}
      {result.figures.length > 0 && (
        <div className="card">
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4 flex items-center gap-2">
            <BarChart3 size={14} className="text-janus-400" />
            Validation Figures
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {result.figures.map(fname => (
              <FigureCard key={fname} sessionId={sessionId} filename={fname} />
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

function MetricCard({ label, value, sub, color }: {
  label: string; value: string; sub: string; color: string
}) {
  const colors: Record<string, string> = {
    blue:   'text-blue-300',
    purple: 'text-purple-300',
    green:  'text-bio-green',
    orange: 'text-bio-gold',
  }
  return (
    <div className="metric-card">
      <div className={`text-2xl font-bold ${colors[color] ?? 'text-white'}`}>{value}</div>
      <div className="text-white text-sm font-medium mt-1">{label}</div>
      <div className="text-gray-600 text-xs mt-0.5">{sub}</div>
    </div>
  )
}

function FigureCard({ sessionId, filename }: { sessionId: string; filename: string }) {
  const url = figureUrl(sessionId, filename)
  const label = filename
    .replace(/_/g, ' ')
    .replace('.png', '')
    .replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className="bg-gray-800/40 border border-gray-800 rounded-xl overflow-hidden">
      <div className="px-3 py-2 flex items-center justify-between border-b border-gray-800">
        <span className="text-xs text-gray-400 font-medium">{label}</span>
        <a
          href={url}
          download={filename}
          className="text-janus-400 hover:text-janus-300 transition-colors"
          title="Download figure"
        >
          <Download size={14} />
        </a>
      </div>
      <img
        src={url}
        alt={label}
        className="w-full h-auto"
        onError={e => { (e.target as HTMLImageElement).style.display = 'none' }}
      />
    </div>
  )
}
