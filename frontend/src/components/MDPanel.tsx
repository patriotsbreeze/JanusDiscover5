import { useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, BarChart3, Download, FlaskConical, Timer } from 'lucide-react'
import { MDResult, MDRequest, figureUrl } from '../api'

interface MDSetupProps {
  sessionId: string
  runMode: 'mock' | 'real'
  onSubmit: (req: Omit<MDRequest, 'session_id'>) => void
}

export function MDSetupPanel({ sessionId, runMode, onSubmit }: MDSetupProps) {
  const [runMD, setRunMD] = useState(true)
  const [value, setValue] = useState(10)
  const [unit, setUnit] = useState<'ps' | 'ns'>('ns')

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      <div className="card border-janus-700 glow-blue">
        <h3 className="font-semibold text-white text-lg mb-1 flex items-center gap-2">
          <Activity size={20} className="text-janus-400" />
          Molecular Dynamics Simulation
        </h3>
        <p className="text-gray-400 text-sm">
          Validate the top-ranked hit's stability in the binding pocket via all-atom MD
          simulation (OpenMM 8, AMBER ff14SB, TIP3P explicit solvent).
        </p>
      </div>

      <div className="card">
        <div className="space-y-4">
          {/* Run MD toggle */}
          <div className="flex items-center justify-between p-4 rounded-xl bg-gray-800/40 border border-gray-800">
            <div>
              <p className="font-medium text-white">Run MD Simulation</p>
              <p className="text-gray-500 text-sm">Adds RMSD, RMSF, Rg, energy figures + updated manuscript</p>
            </div>
            <button
              onClick={() => setRunMD(v => !v)}
              className={`w-14 h-7 rounded-full transition-colors relative ${runMD ? 'bg-janus-600' : 'bg-gray-700'}`}
            >
              <span className={`absolute top-0.5 w-6 h-6 rounded-full bg-white shadow transition-transform ${runMD ? 'left-7' : 'left-0.5'}`} />
            </button>
          </div>

          {runMD && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="overflow-hidden"
            >
              <div className="p-4 rounded-xl bg-gray-800/40 border border-gray-800">
                <p className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
                  <Timer size={14} className="text-janus-400" />
                  Simulation Duration
                </p>
                <div className="flex gap-3 items-center">
                  <input
                    type="number"
                    min={1}
                    max={unit === 'ns' ? 1000 : 1000000}
                    value={value}
                    onChange={e => setValue(Number(e.target.value))}
                    className="input-field w-32"
                  />
                  <div className="flex rounded-xl overflow-hidden border border-gray-700">
                    {(['ps', 'ns'] as const).map(u => (
                      <button
                        key={u}
                        onClick={() => setUnit(u)}
                        className={`px-4 py-2 text-sm font-medium transition-colors ${
                          unit === u ? 'bg-janus-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                        }`}
                      >
                        {u}
                      </button>
                    ))}
                  </div>
                </div>
                <p className="text-gray-600 text-xs mt-2">
                  {unit === 'ns' ? `≈ ${(value * 500_000).toLocaleString()} simulation steps (2 fs)` : `≈ ${(value * 500).toLocaleString()} simulation steps (2 fs)`}
                </p>
                {runMode === 'real' && (
                  <div className="mt-3 p-3 rounded-lg bg-orange-950/30 border border-orange-900">
                    <p className="text-orange-300 text-xs">
                      Real mode: GPU-accelerated via CUDA if available. Long simulations may take significant time.
                    </p>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </div>
      </div>

      <button
        className="btn-primary w-full"
        onClick={() => onSubmit({
          run_md: runMD,
          duration: runMD ? { value, unit } : undefined,
          run_mode: runMode,
        })}
      >
        {runMD ? `▶ Run ${value} ${unit} MD Simulation` : 'Skip MD — Finish'}
      </button>
    </motion.div>
  )
}

interface MDResultsProps {
  result: MDResult
  sessionId: string
}

export function MDResultsPanel({ result, sessionId }: MDResultsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      <div className="card glow-green border-green-900">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="badge bg-green-900/40 text-green-300 border-green-800">MD COMPLETE</span>
              {result.mock && <span className="badge bg-yellow-900/40 text-yellow-300 border-yellow-800">MOCK</span>}
            </div>
            <h3 className="font-semibold text-white text-lg">
              {result.duration_ns} ns Simulation
            </h3>
            <p className="text-gray-400 text-sm">{result.frames_analyzed.toLocaleString()} frames analyzed</p>
          </div>
        </div>
      </div>

      {/* MD metrics */}
      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
          Simulation Statistics
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <MDMetric label="RMSD (mean)" value={`${(result.rmsd_mean_nm * 10).toFixed(2)} Å`} />
          <MDMetric label="RMSD (σ)" value={`${(result.rmsd_std_nm * 10).toFixed(2)} Å`} />
          <MDMetric label="RMSF (mean)" value={`${(result.rmsf_mean_nm * 10).toFixed(2)} Å`} />
          <MDMetric label="Radius of Gyration" value={`${(result.radius_of_gyration_mean_nm * 10).toFixed(2)} Å`} />
          <MDMetric label="Potential Energy" value={`${(result.potential_energy_mean_kj_mol / 1000).toFixed(1)} ×10³ kJ/mol`} />
          <MDMetric label="Convergence" value={result.rmsd_mean_nm < 0.2 ? 'Stable ✓' : 'Check RMSD'} />
        </div>
      </div>

      {/* MD Figures */}
      {result.figures.length > 0 && (
        <div className="card">
          <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4 flex items-center gap-2">
            <BarChart3 size={14} className="text-janus-400" />
            MD Validation Figures
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {result.figures.map(fname => (
              <MDFigureCard key={fname} sessionId={sessionId} filename={fname} />
            ))}
          </div>
        </div>
      )}
    </motion.div>
  )
}

function MDMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <div className="text-white font-semibold">{value}</div>
      <div className="text-gray-500 text-xs mt-1">{label}</div>
    </div>
  )
}

function MDFigureCard({ sessionId, filename }: { sessionId: string; filename: string }) {
  const url = figureUrl(sessionId, filename)
  const label = filename.replace(/_/g, ' ').replace('.png', '').replace(/\b\w/g, c => c.toUpperCase())

  return (
    <div className="bg-gray-800/40 border border-gray-800 rounded-xl overflow-hidden">
      <div className="px-3 py-2 flex items-center justify-between border-b border-gray-800">
        <span className="text-xs text-gray-400 font-medium">{label}</span>
        <a href={url} download={filename} className="text-janus-400 hover:text-janus-300 transition-colors">
          <Download size={14} />
        </a>
      </div>
      <img src={url} alt={label} className="w-full h-auto"
        onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
    </div>
  )
}
