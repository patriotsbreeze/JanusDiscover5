interface Props {
  pct: number
  message?: string
  color?: 'blue' | 'green'
}

export function ProgressBar({ pct, message, color = 'blue' }: Props) {
  const barClass = color === 'green'
    ? 'bg-bio-green'
    : 'progress-shimmer'

  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-2">
        <span className="text-gray-400">{message ?? 'Processing…'}</span>
        <span className="text-gray-300 font-mono">{pct}%</span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${barClass}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
    </div>
  )
}
