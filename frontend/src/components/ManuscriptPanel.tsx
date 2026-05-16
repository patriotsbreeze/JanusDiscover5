import { motion } from 'framer-motion'
import { Download, FileText, ScrollText } from 'lucide-react'
import { ManuscriptResult, manuscriptUrl } from '../api'

interface Props {
  result: ManuscriptResult
  sessionId: string
}

export function ManuscriptPanel({ result, sessionId }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-5"
    >
      <div className="card border-janus-700 glow-blue">
        <h3 className="font-semibold text-white text-lg mb-1 flex items-center gap-2">
          <ScrollText size={20} className="text-janus-400" />
          Manuscript Ready
        </h3>
        <p className="text-gray-400 text-sm">
          A publication-quality LaTeX manuscript (JMLR/ICML level) has been generated with
          all methods, validation results, and figures.
        </p>
      </div>

      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-4">
          Download Files
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <DownloadCard
            title="LaTeX Source"
            desc="manuscript.tex — compile with pdflatex"
            icon={FileText}
            href={manuscriptUrl(sessionId, 'tex')}
            color="blue"
          />
          <DownloadCard
            title="Bibliography"
            desc="refs.bib — 14 real citations"
            icon={ScrollText}
            href={manuscriptUrl(sessionId, 'bib')}
            color="purple"
          />
          {result.pdf_path && (
            <DownloadCard
              title="PDF Manuscript"
              desc="Compiled PDF — ready to submit"
              icon={Download}
              href={manuscriptUrl(sessionId, 'pdf')}
              color="green"
            />
          )}
        </div>
        {!result.pdf_path && (
          <div className="mt-3 p-3 rounded-xl bg-yellow-950/30 border border-yellow-900">
            <p className="text-yellow-300 text-xs">
              PDF compilation requires <code className="font-mono">pdflatex</code> installed on
              the server. Download the <code>.tex</code> source and compile locally with:
              <code className="block mt-1 bg-gray-900 rounded p-2 text-gray-300">
                pdflatex manuscript.tex && pdflatex manuscript.tex
              </code>
            </p>
          </div>
        )}
      </div>

      <div className="card">
        <h4 className="text-sm font-semibold text-gray-300 uppercase tracking-wide mb-3">
          Manuscript Contents
        </h4>
        <ol className="space-y-1.5 text-sm text-gray-400">
          {[
            'Abstract',
            'Introduction — drug discovery landscape & JanusDiscover motivation',
            'Methods — LBDD (ECFP4 + RF), SBDD (AutoDock Vina 1.2), hybrid consensus',
            'Methods — MD simulation (OpenMM 8, ff14SB, TIP3P)',
            'Validation metrics — AUC-ROC, BEDROC, EF1%, EF5%, redocking RMSD',
            'Results and Discussion — metrics table + all figures',
            'Conclusion & future directions',
            'References — 14 real peer-reviewed citations',
          ].map((item, i) => (
            <li key={i} className="flex gap-2">
              <span className="text-gray-600 w-5 flex-shrink-0">{i + 1}.</span>
              {item}
            </li>
          ))}
        </ol>
      </div>
    </motion.div>
  )
}

function DownloadCard({ title, desc, icon: Icon, href, color }: {
  title: string; desc: string; icon: any; href: string; color: string
}) {
  const colors: Record<string, string> = {
    blue:   'bg-janus-900/40 border-janus-800 hover:border-janus-600',
    purple: 'bg-purple-900/40 border-purple-800 hover:border-purple-600',
    green:  'bg-green-900/40 border-green-800 hover:border-green-600',
  }
  const iconColors: Record<string, string> = {
    blue: 'text-janus-400', purple: 'text-purple-400', green: 'text-bio-green',
  }
  return (
    <a
      href={href}
      download
      className={`flex flex-col gap-3 p-4 rounded-xl border transition-all ${colors[color]}`}
    >
      <Icon size={24} className={iconColors[color]} />
      <div>
        <p className="font-medium text-white text-sm">{title}</p>
        <p className="text-gray-500 text-xs mt-0.5">{desc}</p>
      </div>
      <div className="flex items-center gap-1 text-xs text-gray-500 mt-auto">
        <Download size={12} /> Download
      </div>
    </a>
  )
}
