"use client"
import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

interface Candidate {
  tic_id: number
  orbital_period_days: number
  transit_epoch_btjd: number
  transit_depth_ppm: number
  planet_radius_rearth: number
  planet_radius_rjup: number
  impact_parameter: number
  transit_duration_hours: number
  semi_major_axis_au: number
  equilibrium_temperature_k: number
  insolation_earth: number
  stellar_teff: number
  stellar_radius: number
  stellar_mass: number
  red_chi2: number
  quality_flag: string
  class_name: string
  atmosphere_class: string
  planet_density_gcc: number
  transit_probability: number
}

export default function CandidateDetail() {
  const params = useParams()
  const ticId = Number(params.id)
  const [c, setC] = useState<Candidate | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/api/candidates")
      .then((r) => r.json())
      .then((data) => {
        const list: Candidate[] = Array.isArray(data) ? data : []
        setC(list.find((x) => x.tic_id === ticId) || null)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [ticId])

  if (loading) return <div className="space-y-4"><div className="h-8 w-48 skeleton" /><div className="h-64 skeleton" /></div>

  if (!c) {
    return (
      <div className="py-24 text-center space-y-4">
        <p className="text-lg">Candidate TIC {ticId} not found</p>
        <Link href="/candidates" className="text-sm underline" style={{ color: 'var(--color-muted)' }}>Back to candidates</Link>
      </div>
    )
  }

  const figs = [
    { name: "Phase-folded", src: `/figures/phase_fold_${ticId}.png` },
    { name: "Transit Fit", src: `/figures/transit_fit_pub_${ticId}.png` },
    { name: "Periodogram", src: `/figures/periodogram_${ticId}.png` },
    { name: "Publication Panel", src: `/figures/publication_panel_${ticId}.png` },
    { name: "Residuals", src: `/figures/residual_dist_${ticId}.png` },
    { name: "Raw Light Curve", src: `/figures/raw_lc_${ticId}.png` },
  ]

  const paramsList = [
    { label: "Orbital Period", value: c.orbital_period_days, unit: "days" },
    { label: "Transit Epoch", value: c.transit_epoch_btjd, unit: "BTJD" },
    { label: "Planet Radius", value: c.planet_radius_rearth, unit: "R⊕" },
    { label: "Radius (Jupiter)", value: c.planet_radius_rjup, unit: "R<sub>J</sub>" },
    { label: "Transit Depth", value: c.transit_depth_ppm, unit: "ppm" },
    { label: "Duration", value: c.transit_duration_hours, unit: "hr" },
    { label: "Semi-major Axis", value: c.semi_major_axis_au, unit: "AU" },
    { label: "Impact Parameter", value: c.impact_parameter, unit: "" },
    { label: "Equilibrium T<sub>eq</sub>", value: c.equilibrium_temperature_k, unit: "K" },
    { label: "Stellar T<sub>eff</sub>", value: c.stellar_teff, unit: "K" },
    { label: "Stellar Radius", value: c.stellar_radius, unit: "R☉" },
    { label: "Stellar Mass", value: c.stellar_mass, unit: "M☉" },
    { label: "Density", value: c.planet_density_gcc, unit: "g/cm³" },
    { label: "χ²<sub>red</sub>", value: c.red_chi2, unit: "" },
    { label: "Insolation", value: c.insolation_earth, unit: "Earth" },
    { label: "Transit Prob.", value: c.transit_probability, unit: "%" },
  ]

  return (
    <div className="space-y-12 max-w-5xl">
      <div className="flex items-center gap-4">
        <Link href="/candidates" className="p-1 -ml-1 transition-opacity hover:opacity-60">
          <ArrowLeft size={18} />
        </Link>
        <div>
          <h1 className="text-3xl font-light tracking-tight">TIC {c.tic_id}</h1>
          <div className="flex items-center gap-3 mt-1.5">
            <span
              className="text-xs font-medium px-2.5 py-0.5 rounded-full"
              style={{
                background: c.quality_flag === "PASS" ? 'hsl(142, 50%, 90%)' : 'hsl(38, 92%, 90%)',
                color: c.quality_flag === "PASS" ? 'hsl(142, 50%, 30%)' : 'hsl(38, 92%, 30%)',
              }}
            >
              {c.quality_flag}
            </span>
            <span className="text-xs" style={{ color: 'var(--color-muted)' }}>{c.class_name}</span>
            <span className="text-xs" style={{ color: 'var(--color-muted)' }}>{c.atmosphere_class}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-px" style={{ background: 'var(--color-border)' }}>
        {paramsList.map((p) => (
          <div key={p.label} className="p-4 lg:p-5" style={{ background: 'var(--color-bg)' }}>
            <p className="text-xs mb-1.5" style={{ color: 'var(--color-muted)' }} dangerouslySetInnerHTML={{ __html: p.label }} />
            <p className="text-base font-medium">
              {p.value != null && !isNaN(p.value)
                ? p.value < 0.01 ? p.value.toExponential(2) : p.value.toFixed(p.value < 1 ? 4 : 2)
                : "—"}
              {p.value != null && p.unit && <span className="text-xs ml-1" style={{ color: 'var(--color-muted)' }} dangerouslySetInnerHTML={{ __html: p.unit }} />}
            </p>
          </div>
        ))}
      </div>

      <div className="space-y-8">
        <h2 className="text-sm font-medium">Figures</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {figs.map((fig) => (
            <div key={fig.name} className="space-y-2">
              <p className="text-xs" style={{ color: 'var(--color-muted)' }}>{fig.name}</p>
              <div className="rounded-lg overflow-hidden" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
                <img src={fig.src} alt={fig.name} className="w-full object-contain" loading="lazy"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none" }} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
