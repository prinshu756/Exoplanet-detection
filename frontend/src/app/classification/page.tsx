"use client"
import { useEffect, useState } from "react"

interface Candidate {
  class_name: string
  atmosphere_class: string
  planet_radius_rearth: number
}

export default function ClassificationPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/api/candidates")
      .then((r) => r.json())
      .then((data) => { setCandidates(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  if (loading) return <div className="space-y-4"><div className="h-8 w-48 skeleton" /><div className="h-64 skeleton" /></div>

  const classDist: Record<string, number> = {}
  candidates.forEach((c) => { const k = c.class_name || "Unknown"; classDist[k] = (classDist[k] || 0) + 1 })

  const atmDist: Record<string, number> = {}
  candidates.forEach((c) => { const k = c.atmosphere_class || "Unknown"; atmDist[k] = (atmDist[k] || 0) + 1 })

  const sizeDist: Record<string, number> = {}
  candidates.forEach((c) => {
    const r = c.planet_radius_rearth
    let size = "Unknown"
    if (r != null) {
      if (r < 1) size = "Sub-Earth"
      else if (r < 1.5) size = "Earth-size"
      else if (r < 3) size = "Super-Earth"
      else if (r < 6) size = "Sub-Neptune"
      else if (r < 12) size = "Neptune-size"
      else size = "Jupiter-size"
    }
    sizeDist[size] = (sizeDist[size] || 0) + 1
  })

  const pipeline = [
    { step: "01", title: "Feature Extraction", desc: "50 features from light curves, periodograms, transit parameters" },
    { step: "02", title: "Random Forest", desc: "500 trees · 4 classes (Planet, EB, BEB, Stellar) · 61.1% accuracy" },
    { step: "03", title: "Neural Network", desc: "50→256→128→64→32→4 · 57K params · 50.0% accuracy" },
    { step: "04", title: "Atmosphere Classifier", desc: "Density-based: Gas-rich, Extended, Rocky, Iron-rich" },
  ]

  return (
    <div className="space-y-12 max-w-5xl">
      <div className="space-y-2">
        <h1 className="text-3xl font-light tracking-tight">Classification</h1>
        <p className="text-sm" style={{ color: 'var(--color-muted)' }}>ML-based candidate classification and atmospheric characterization</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-px" style={{ background: 'var(--color-border)' }}>
        {[
          { label: "Model Accuracy (RF)", value: "61.1%" },
          { label: "Predicted Planets", value: String(classDist["Planet"] || 0) },
          { label: "Atmosphere Classes", value: String(Object.keys(atmDist).length) },
        ].map((s) => (
          <div key={s.label} className="p-6" style={{ background: 'var(--color-bg)' }}>
            <p className="text-xs mb-1.5" style={{ color: 'var(--color-muted)' }}>{s.label}</p>
            <p className="text-2xl font-light">{s.value}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="space-y-4">
          <h2 className="text-sm font-medium">Class Distribution</h2>
          <div className="space-y-2">
            {Object.entries(classDist).sort(([, a], [, b]) => b - a).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between h-9 px-3 rounded-lg text-sm" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
                <span>{k}</span>
                <span className="font-medium">{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="space-y-4">
          <h2 className="text-sm font-medium">Atmosphere Types</h2>
          <div className="space-y-2">
            {Object.entries(atmDist).sort(([, a], [, b]) => b - a).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between h-9 px-3 rounded-lg text-sm" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
                <span>{k}</span>
                <span className="font-medium">{v}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="space-y-4">
          <h2 className="text-sm font-medium">Size Classes</h2>
          <div className="space-y-2">
            {Object.entries(sizeDist).sort(([, a], [, b]) => b - a).map(([k, v]) => (
              <div key={k} className="flex items-center justify-between h-9 px-3 rounded-lg text-sm" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
                <span>{k}</span>
                <span className="font-medium">{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-6">
        <h2 className="text-sm font-medium">Pipeline</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {pipeline.map((p) => (
            <div key={p.step} className="space-y-2">
              <p className="text-xs" style={{ color: 'var(--color-muted)' }}>{p.step}</p>
              <h3 className="text-base font-medium">{p.title}</h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--color-muted)' }}>{p.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
