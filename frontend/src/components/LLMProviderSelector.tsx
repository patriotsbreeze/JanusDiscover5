import { useState, useEffect } from 'react'
import { Eye, EyeOff, ChevronDown } from 'lucide-react'
import type { LLMProvider } from '../api'

interface ProviderConfig {
  label: string
  placeholder: string
  envHint: string
  defaultModel: string
  modelOptions: string[]
  needsKey: boolean
  color: string
}

const PROVIDERS: Record<LLMProvider, ProviderConfig> = {
  anthropic: {
    label: 'Anthropic (Claude)',
    placeholder: 'sk-ant-…',
    envHint: 'ANTHROPIC_API_KEY',
    defaultModel: 'claude-opus-4-7',
    modelOptions: ['claude-opus-4-7', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'],
    needsKey: true,
    color: 'text-orange-300',
  },
  openai: {
    label: 'OpenAI (ChatGPT)',
    placeholder: 'sk-…',
    envHint: 'OPENAI_API_KEY',
    defaultModel: 'gpt-4o',
    modelOptions: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'o1'],
    needsKey: true,
    color: 'text-green-300',
  },
  google: {
    label: 'Google Gemini',
    placeholder: 'AIza…',
    envHint: 'GOOGLE_API_KEY',
    defaultModel: 'gemini-2.0-flash',
    modelOptions: ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'],
    needsKey: true,
    color: 'text-blue-300',
  },
  deepseek: {
    label: 'DeepSeek',
    placeholder: 'sk-…',
    envHint: 'DEEPSEEK_API_KEY',
    defaultModel: 'deepseek-chat',
    modelOptions: ['deepseek-chat', 'deepseek-reasoner'],
    needsKey: true,
    color: 'text-purple-300',
  },
  ollama: {
    label: 'Ollama (local Llama)',
    placeholder: '(no key needed)',
    envHint: '',
    defaultModel: 'llama3.2',
    modelOptions: ['llama3.2', 'llama3.1', 'mistral', 'mixtral', 'phi3'],
    needsKey: false,
    color: 'text-cyan-300',
  },
}

const STORAGE_KEY = 'jd_llm_settings'

interface LLMSettings {
  provider: LLMProvider
  apiKeys: Partial<Record<LLMProvider, string>>
  models: Partial<Record<LLMProvider, string>>
}

function loadSettings(): LLMSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) return JSON.parse(raw)
  } catch {}
  return { provider: 'anthropic', apiKeys: {}, models: {} }
}

function saveSettings(s: LLMSettings) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(s)) } catch {}
}

interface Props {
  disabled?: boolean
  onChange: (provider: LLMProvider, apiKey: string, model: string) => void
}

export function LLMProviderSelector({ disabled, onChange }: Props) {
  const [settings, setSettings] = useState<LLMSettings>(loadSettings)
  const [showKey, setShowKey] = useState(false)
  const [expanded, setExpanded] = useState(false)

  const provider = settings.provider
  const cfg = PROVIDERS[provider]
  const apiKey = settings.apiKeys[provider] ?? ''
  const model = settings.models[provider] ?? cfg.defaultModel

  useEffect(() => {
    saveSettings(settings)
    onChange(provider, apiKey, model)
  }, [provider, apiKey, model])

  const setProvider = (p: LLMProvider) => setSettings(s => ({ ...s, provider: p }))
  const setKey = (k: string) =>
    setSettings(s => ({ ...s, apiKeys: { ...s.apiKeys, [provider]: k } }))
  const setModel = (m: string) =>
    setSettings(s => ({ ...s, models: { ...s.models, [provider]: m } }))

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-300">AI Model (for literature review)</label>
        <button
          type="button"
          onClick={() => setExpanded(e => !e)}
          className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1 transition-colors"
        >
          {expanded ? 'Collapse' : 'Configure'}
          <ChevronDown size={12} className={`transition-transform ${expanded ? 'rotate-180' : ''}`} />
        </button>
      </div>

      {/* Summary chip (always visible) */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-800/60 border border-gray-700 text-sm">
        <span className={`font-semibold ${cfg.color}`}>{cfg.label}</span>
        <span className="text-gray-500">·</span>
        <span className="text-gray-400 font-mono text-xs">{model}</span>
        {cfg.needsKey && (
          <>
            <span className="text-gray-500">·</span>
            <span className={`text-xs ${apiKey ? 'text-bio-green' : 'text-yellow-400'}`}>
              {apiKey ? 'Key saved' : 'No key'}
            </span>
          </>
        )}
        {!cfg.needsKey && <span className="text-xs text-cyan-400">local</span>}
      </div>

      {expanded && (
        <div className="p-4 rounded-xl border border-gray-700 bg-gray-900/60 space-y-4">
          {/* Provider pills */}
          <div>
            <p className="text-xs text-gray-500 mb-2">Provider</p>
            <div className="flex flex-wrap gap-2">
              {(Object.keys(PROVIDERS) as LLMProvider[]).map(p => (
                <button
                  key={p}
                  type="button"
                  disabled={disabled}
                  onClick={() => setProvider(p)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    provider === p
                      ? 'border-janus-600 bg-janus-950/60 ' + PROVIDERS[p].color
                      : 'border-gray-700 bg-gray-800 text-gray-400 hover:border-gray-600'
                  }`}
                >
                  {PROVIDERS[p].label}
                </button>
              ))}
            </div>
          </div>

          {/* Model picker */}
          <div>
            <p className="text-xs text-gray-500 mb-2">Model</p>
            <div className="flex gap-2">
              <select
                value={model}
                onChange={e => setModel(e.target.value)}
                disabled={disabled}
                className="flex-1 input-field text-sm py-2"
              >
                {cfg.modelOptions.map(m => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
              <input
                value={model}
                onChange={e => setModel(e.target.value)}
                disabled={disabled}
                placeholder="or type custom model ID"
                className="flex-1 input-field text-sm py-2 font-mono"
              />
            </div>
          </div>

          {/* API key */}
          {cfg.needsKey && (
            <div>
              <p className="text-xs text-gray-500 mb-2">
                API Key
                {cfg.envHint && (
                  <span className="ml-1 text-gray-600">
                    (or set <code className="text-gray-500">{cfg.envHint}</code> env var on the server)
                  </span>
                )}
              </p>
              <div className="relative">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={apiKey}
                  onChange={e => setKey(e.target.value)}
                  disabled={disabled}
                  placeholder={cfg.placeholder}
                  className="input-field w-full pr-10 font-mono text-sm"
                />
                <button
                  type="button"
                  onClick={() => setShowKey(v => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                  tabIndex={-1}
                >
                  {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              <p className="text-xs text-gray-600 mt-1">
                Stored in your browser only — never sent anywhere except the backend for this session.
              </p>
            </div>
          )}

          {provider === 'ollama' && (
            <p className="text-xs text-yellow-600/80">
              Make sure Ollama is running locally (<code>ollama serve</code>) and the model is pulled
              (<code>ollama pull {model}</code>). Set <code>OLLAMA_BASE_URL</code> on the server if it's not at <code>localhost:11434</code>.
            </p>
          )}
        </div>
      )}
    </div>
  )
}
