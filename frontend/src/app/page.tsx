"use client"
import { useEffect, useState } from "react"
import Link from "next/link"
import { Search, ArrowUpRight, Upload, FileDown } from "lucide-react"

interface Candidate {
  tic_id: number
  planet_radius_rearth: number
  orbital_period_days: number
  equilibrium_temperature_k: number
  red_chi2: number
  quality_flag: string
  class_name: string
}

export default function Dashboard() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState("")
  const [showUpload, setShowUpload] = useState(false)

  useEffect(() => {
    fetch("/api/candidates")
      .then((r) => r.json())
      .then((data) => { setCandidates(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const filtered = candidates.filter((c) =>
    String(c.tic_id).includes(searchQuery)
  ).slice(0, 6)

  const stats = [
    { label: "Candidates", value: candidates.length },
    { label: "Confirmed", value: candidates.filter((c) => c.quality_flag === "PASS").length },
    { label: "Median Period", value: candidates.length ? candidates.reduce((s, c) => s + c.orbital_period_days, 0) / candidates.length : 0, unit: "days" },
    { label: "Median Radius", value: candidates.length ? candidates.reduce((s, c) => s + c.planet_radius_rearth, 0) / candidates.length : 0, unit: "R⊕" },
  ]

  return (
    <div className="space-y-16">
      <div className="space-y-6 max-w-2xl">
        <h1 className="text-4xl sm:text-5xl font-light tracking-tight" style={{ color: 'var(--color-fg)' }}>
          Exoplanet<br />Detection
        </h1>
        <p className="text-base leading-relaxed" style={{ color: 'var(--color-muted)' }}>
          AI-powered transit analysis pipeline. Search candidates by TIC ID or upload light curve data for real-time classification and parameter estimation.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row gap-3 max-w-xl">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={16} style={{ color: 'var(--color-muted)' }} />
          <input
            type="text"
            placeholder="Search by TIC ID…"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full h-12 pl-10 pr-4 rounded-lg text-sm border transition-all duration-200 focus:outline-none"
            style={{
              background: 'var(--color-card)',
              borderColor: searchQuery ? 'var(--color-fg)' : 'var(--color-border)',
              color: 'var(--color-fg)',
            }}
          />
        </div>
        <button
          onClick={() => setShowUpload(!showUpload)}
          className="h-12 px-5 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2"
          style={{ background: 'var(--color-fg)', color: 'var(--color-card)' }}
        >
          <Upload size={16} />
          Upload
        </button>
      </div>

      {searchQuery && (
        <div className="space-y-2 -mt-8">
          {loading ? (
            <div className="h-12 skeleton w-full" />
          ) : filtered.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--color-muted)' }}>No candidates found for TIC {searchQuery}</p>
          ) : (
            filtered.map((c) => (
              <Link
                key={c.tic_id}
                href={`/candidates/${c.tic_id}`}
                className="flex items-center justify-between h-12 px-4 rounded-lg text-sm transition-all duration-200 group"
                style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}
              >
                <span className="font-medium">TIC {c.tic_id}</span>
                <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--color-muted)' }}>
                  <span>{c.planet_radius_rearth?.toFixed(2)} R⊕</span>
                  <span>{c.orbital_period_days?.toFixed(2)} d</span>
                  <span>{c.class_name || "—"}</span>
                  <ArrowUpRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--color-fg)' }} />
                </div>
              </Link>
            ))
          )}
        </div>
      )}

      {showUpload && (
        <div
          className="rounded-lg border p-8 space-y-4 fade-in"
          style={{ background: 'var(--color-card)', borderColor: 'var(--color-border)' }}
        >
          <h2 className="text-sm font-medium">Upload Light Curve</h2>
          <p className="text-xs" style={{ color: 'var(--color-muted)' }}>
            Supported formats: Parquet (.parquet), FITS (.fits), CSV (.csv)
          </p>
          <div
            className="border-2 border-dashed rounded-lg p-12 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors"
            style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg)' }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault()
              const file = e.dataTransfer.files[0]
              if (file) handleUpload(file)
            }}
          >
            <FileDown size={24} style={{ color: 'var(--color-muted)' }} />
            <p className="text-sm" style={{ color: 'var(--color-muted)' }}>
              Drop a file here or <span className="underline" style={{ color: 'var(--color-fg)' }}>browse</span>
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-px" style={{ background: 'var(--color-border)' }}>
        {stats.map((s) => (
          <div key={s.label} className="p-6 lg:p-8" style={{ background: 'var(--color-bg)' }}>
            <p className="text-xs font-medium mb-2" style={{ color: 'var(--color-muted)' }}>{s.label}</p>
            <p className="text-2xl font-light">
              {typeof s.value === "number" ? s.value.toFixed(s.value < 1 ? 3 : 1) : s.value}
              {s.unit && <span className="text-sm ml-1" style={{ color: 'var(--color-muted)' }}>{s.unit}</span>}
            </p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="space-y-4">
          <h2 className="text-sm font-medium">Radius-Period Diagram</h2>
          <div className="aspect-[4/3] rounded-lg overflow-hidden" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
            <img src="/figures/radius_period_diagram.png" alt="Radius-Period" className="w-full h-full object-contain" loading="lazy" />
          </div>
        </div>
        <div className="space-y-4">
          <h2 className="text-sm font-medium">Temperature-Insolation</h2>
          <div className="aspect-[4/3] rounded-lg overflow-hidden" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
            <img src="/figures/temp_insolation_diagram.png" alt="Temperature-Insolation" className="w-full h-full object-contain" loading="lazy" />
          </div>
        </div>
      </div>
    </div>
  )
}

async function handleUpload(file: File) {
  const formData = new FormData()
  formData.append("file", file)
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: formData })
    if (!res.ok) throw new Error("Analysis failed")
    const result = await res.json()
    console.log("Analysis result:", result)
  } catch (e) {
    console.error("Upload failed:", e)
  }
}
