export interface Candidate {
  orbital_period_days: number
  transit_epoch_btjd: number
  transit_depth: number
  transit_depth_ppm: number
  radius_ratio: number
  planet_radius_rsun: number
  planet_radius_rearth: number
  planet_radius_rjup: number
  impact_parameter: number
  transit_duration_hours: number
  semi_major_axis_rs: number
  semi_major_axis_au: number
  equilibrium_temperature_k: number
  transit_probability: number
  insolation_earth: number
  tic_id: number
  stellar_radius: number
  stellar_teff: number
  stellar_mass: number
  red_chi2: number
  quality_flag: string
  class_name: string
  planet_density_gcc: number
  atmosphere_class: string
}

export interface SummaryStats {
  n_candidates: number
  n_confirmed: number
  n_promising: number
  n_uncertain: number
  median_period: number
  median_radius: number
  median_teq: number
  median_chi2: number
  median_quality: number
  n_mcmc: number
}

export interface AnalysisReport {
  timestamp: string
  n_candidates: number
  n_validated: number
  n_promising: number
  top_candidates: { tic_id: number; quality_score: number }[]
  fit_statistics: { median_red_chi2: number; median_r2: number }
}

export interface FigureInfo {
  name: string
  path: string
  label: string
  category: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api"

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
  return res.json()
}

export async function getCandidates(): Promise<Candidate[]> {
  return fetchJson<Candidate[]>(`${API_BASE}/candidates`)
}

export async function getCandidate(id: number): Promise<Candidate | undefined> {
  const candidates = await getCandidates()
  return candidates.find((c) => c.tic_id === id)
}

export async function getSummaryStats(): Promise<SummaryStats> {
  return fetchJson<SummaryStats>(`${API_BASE}/summary`)
}

export async function getAnalysisReport(): Promise<AnalysisReport> {
  return fetchJson<AnalysisReport>(`${API_BASE}/analysis-report`)
}

export function getFigures(): FigureInfo[] {
  const figures: FigureInfo[] = [
    { name: "radius_period_diagram", path: "/figures/radius_period_diagram.png", label: "Radius-Period Diagram", category: "Diagrams" },
    { name: "temp_insolation_diagram", path: "/figures/temp_insolation_diagram.png", label: "Temperature-Insolation", category: "Diagrams" },
    { name: "period_vs_duration", path: "/figures/period_vs_duration.png", label: "Period vs Duration", category: "Diagrams" },
    { name: "snr_vs_depth", path: "/figures/snr_vs_depth.png", label: "SNR vs Depth", category: "Analysis" },
    { name: "teff_vs_radius", path: "/figures/teff_vs_radius.png", label: "T<sub>eff</sub> vs Radius", category: "Analysis" },
    { name: "size_distribution_bar", path: "/figures/size_distribution_bar.png", label: "Size Distribution", category: "Statistics" },
    { name: "parameter_correlations", path: "/figures/parameter_correlations.png", label: "Parameter Correlations", category: "Statistics" },
    { name: "quality_scores", path: "/figures/quality_scores.png", label: "Quality Scores", category: "Statistics" },
    { name: "optimization_summary", path: "/figures/optimization_summary.png", label: "Optimization Summary", category: "Analysis" },
    { name: "atmosphere_classification", path: "/figures/atmosphere_classification.png", label: "Atmosphere Classes", category: "Classification" },
    { name: "habitability_diagram", path: "/figures/habitability_diagram.png", label: "Habitability Diagram", category: "Diagrams" },
    { name: "all_transits_comparison", path: "/figures/all_transits_comparison.png", label: "All Transits", category: "Transits" },
  ]
  return figures
}

export function getCandidateFigures(ticId: number): FigureInfo[] {
  return [
    { name: `phase_fold_${ticId}`, path: `/figures/phase_fold_${ticId}.png`, label: "Phase-folded Light Curve", category: "Light Curve" },
    { name: `transit_fit_pub_${ticId}`, path: `/figures/transit_fit_pub_${ticId}.png`, label: "Transit Fit", category: "Transit" },
    { name: `periodogram_${ticId}`, path: `/figures/periodogram_${ticId}.png`, label: "Periodogram", category: "Period" },
    { name: `batman_comparison_${ticId}`, path: `/figures/batman_comparison_${ticId}.png`, label: "Batman Model", category: "Model" },
    { name: `parameter_table_${ticId}`, path: `/figures/parameter_table_${ticId}.png`, label: "Parameters Table", category: "Parameters" },
    { name: `publication_panel_${ticId}`, path: `/figures/publication_panel_${ticId}.png`, label: "Publication Panel", category: "Publication" },
    { name: `residual_dist_${ticId}`, path: `/figures/residual_dist_${ticId}.png`, label: "Residuals", category: "Residuals" },
    { name: `raw_lc_${ticId}`, path: `/figures/raw_lc_${ticId}.png`, label: "Raw Light Curve", category: "Light Curve" },
  ]
}

export const SIZE_CLASSIFICATION = [
  { label: "Sub-Earth", range: "< 1 R<sub>⊕</sub>", color: "#b0b0b0" },
  { label: "Earth-size", range: "1–1.5 R<sub>⊕</sub>", color: "#4caf50" },
  { label: "Super-Earth", range: "1.5–3 R<sub>⊕</sub>", color: "#2196f3" },
  { label: "Sub-Neptune", range: "3–6 R<sub>⊕</sub>", color: "#ff9800" },
  { label: "Neptune-size", range: "6–12 R<sub>⊕</sub>", color: "#f44336" },
  { label: "Jupiter-size", range: "> 12 R<sub>⊕</sub>", color: "#9c27b0" },
]

export function classifySize(radiusRe: number): string {
  if (radiusRe < 1) return "Sub-Earth"
  if (radiusRe < 1.5) return "Earth-size"
  if (radiusRe < 3) return "Super-Earth"
  if (radiusRe < 6) return "Sub-Neptune"
  if (radiusRe < 12) return "Neptune-size"
  return "Jupiter-size"
}
