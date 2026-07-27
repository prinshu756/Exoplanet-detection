"use client"
import { useEffect, useState, useMemo } from "react"
import Link from "next/link"
import { Search, ArrowUpRight } from "lucide-react"

interface Candidate {
  tic_id: number
  planet_radius_rearth: number
  orbital_period_days: number
  equilibrium_temperature_k: number
  red_chi2: number
  quality_flag: string
  class_name: string
  atmosphere_class: string
}

export default function CandidatesPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState("")

  useEffect(() => {
    fetch("/api/candidates")
      .then((r) => r.json())
      .then((data) => { setCandidates(Array.isArray(data) ? data : []); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    if (!search) return candidates
    const q = search.toLowerCase()
    return candidates.filter((c) => String(c.tic_id).includes(q) || c.class_name?.toLowerCase().includes(q))
  }, [candidates, search])

  return (
    <div className="space-y-10 max-w-4xl">
      <div className="space-y-2">
        <h1 className="text-3xl font-light tracking-tight">Candidates</h1>
        <p className="text-sm" style={{ color: 'var(--color-muted)' }}>
          {candidates.length} detected exoplanet candidates
        </p>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={16} style={{ color: 'var(--color-muted)' }} />
        <input
          type="text"
          placeholder="Search by TIC ID or class…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full h-11 pl-10 pr-4 rounded-lg text-sm border transition-all duration-200 focus:outline-none"
          style={{ background: 'var(--color-card)', borderColor: 'var(--color-border)', color: 'var(--color-fg)' }}
        />
      </div>

      {loading ? (
        <div className="space-y-2">
          {Array.from({ length: 8 }).map((_, i) => <div key={i} className="h-14 skeleton w-full" />)}
        </div>
      ) : (
        <div className="space-y-px" style={{ background: 'var(--color-border)' }}>
          {filtered.map((c) => (
            <Link
              key={c.tic_id}
              href={`/candidates/${c.tic_id}`}
              className="flex items-center justify-between h-14 px-4 text-sm transition-all duration-200 group"
              style={{ background: 'var(--color-bg)' }}
            >
              <div className="flex items-center gap-6">
                <span className="font-medium w-28">TIC {c.tic_id}</span>
                <span className="text-xs" style={{ color: 'var(--color-muted)' }}>
                  {c.planet_radius_rearth != null ? `${c.planet_radius_rearth.toFixed(3)} R⊕` : "—"}
                </span>
                <span className="text-xs" style={{ color: 'var(--color-muted)' }}>
                  {c.orbital_period_days?.toFixed(2)} d
                </span>
              </div>
              <div className="flex items-center gap-4">
                <span
                  className="text-xs font-medium px-2.5 py-0.5 rounded-full"
                  style={{
                    background: c.quality_flag === "PASS" ? 'hsl(142, 50%, 90%)' : 'hsl(38, 92%, 90%)',
                    color: c.quality_flag === "PASS" ? 'hsl(142, 50%, 30%)' : 'hsl(38, 92%, 30%)',
                  }}
                >
                  {c.quality_flag}
                </span>
                <ArrowUpRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--color-muted)' }} />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
