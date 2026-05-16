import { motion } from 'framer-motion'
import { FlaskConical, Dna, Zap, BookOpen, BarChart3, Atom } from 'lucide-react'

interface Props { onStart: () => void }

const features = [
  { icon: BookOpen,    title: 'AI Literature Review',    desc: 'LLM-powered synthesis of existing therapies, discovery history & method recommendation' },
  { icon: Dna,         title: 'LBDD Pipeline',           desc: 'ECFP4 fingerprints + Random Forest ensemble with BEDROC / EF1% / AUC-ROC validation' },
  { icon: Atom,        title: 'SBDD Pipeline',           desc: 'AutoDock Vina 1.2 docking with redocking RMSD validation against co-crystal structures' },
  { icon: Zap,         title: 'Hybrid Mode',             desc: 'Consensus LBDD+SBDD scoring for targets with both structural data and known actives' },
  { icon: FlaskConical, title: 'MD Simulations',         desc: 'OpenMM 8 all-atom simulations with RMSD, RMSF, Rg and energy validation figures' },
  { icon: BarChart3,   title: 'Publication Figures',     desc: 'Auto-generated matplotlib/seaborn figures + LaTeX manuscript (JMLR/ICML level)' },
]

export function LandingHero({ onStart }: Props) {
  return (
    <div className="min-h-screen flex flex-col bg-gray-950">
      {/* Header */}
      <nav className="border-b border-gray-800 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-janus-500 to-bio-green flex items-center justify-center">
            <FlaskConical size={16} className="text-white" />
          </div>
          <span className="font-bold text-xl tracking-tight text-white">JanusDiscover</span>
          <span className="badge bg-janus-900/60 text-janus-300 border border-janus-800 ml-2">v1.0</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-gray-400 text-sm">Bridging LBDD & SBDD</span>
          <button onClick={onStart} className="btn-primary">
            Launch App →
          </button>
        </div>
      </nav>

      {/* Hero */}
      <div className="flex-1 flex flex-col items-center justify-center px-6 py-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-janus-950 border border-janus-800 text-janus-300 text-sm mb-8">
            <span className="w-2 h-2 rounded-full bg-bio-green animate-pulse" />
            JMLR / ICML-level computational drug discovery
          </div>

          <h1 className="text-6xl font-bold text-white mb-4 leading-tight">
            Discover Drugs<br />
            <span className="bg-gradient-to-r from-janus-400 to-bio-green bg-clip-text text-transparent">
              Intelligently
            </span>
          </h1>

          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            JanusDiscover unifies ligand-based and structure-based drug discovery into one
            intuitive web platform. From AI literature review to MD simulations — with
            publication-quality validation at every step.
          </p>

          <div className="flex gap-4 justify-center">
            <button onClick={onStart} className="btn-primary text-lg px-8 py-3">
              Start Discovery →
            </button>
            <a
              href="https://github.com/patriotsbreeze/JanusDiscover5"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary text-lg px-8 py-3"
            >
              View on GitHub
            </a>
          </div>
        </motion.div>

        {/* Mode badges */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="flex gap-6 mt-12"
        >
          <div className="flex items-center gap-2 text-gray-400 text-sm">
            <span className="badge bg-yellow-900/40 text-yellow-300 border border-yellow-800">MOCK</span>
            Full workflow demo — no API keys needed
          </div>
          <div className="flex items-center gap-2 text-gray-400 text-sm">
            <span className="badge bg-green-900/40 text-green-300 border border-green-800">REAL</span>
            Live ChEMBL · AutoDock Vina · OpenMM
          </div>
        </motion.div>
      </div>

      {/* Features grid */}
      <div className="px-8 pb-20 max-w-6xl mx-auto w-full">
        <h2 className="text-2xl font-semibold text-white text-center mb-10">
          Complete Drug Discovery Pipeline
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * i, duration: 0.4 }}
              className="card hover:border-janus-700 transition-colors"
            >
              <div className="w-10 h-10 rounded-xl bg-janus-900/60 border border-janus-800 flex items-center justify-center mb-4">
                <f.icon size={20} className="text-janus-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">{f.title}</h3>
              <p className="text-gray-400 text-sm leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-8 py-6 text-center text-gray-600 text-sm">
        JanusDiscover © 2025 · Built with FastAPI, React, RDKit, AutoDock Vina, OpenMM
      </footer>
    </div>
  )
}
