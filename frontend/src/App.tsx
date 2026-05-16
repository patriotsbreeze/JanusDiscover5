import { useState } from 'react'
import { DiscoveryPage } from './pages/DiscoveryPage'
import { LandingHero } from './components/LandingHero'

export default function App() {
  const [started, setStarted] = useState(false)

  if (!started) return <LandingHero onStart={() => setStarted(true)} />
  return <DiscoveryPage />
}
