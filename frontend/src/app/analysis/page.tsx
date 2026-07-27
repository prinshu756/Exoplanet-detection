"use client"

const figures = [
  { label: "Radius-Period", src: "/figures/radius_period_diagram.png", category: "Diagrams" },
  { label: "Temperature-Insolation", src: "/figures/temp_insolation_diagram.png", category: "Diagrams" },
  { label: "Period vs Duration", src: "/figures/period_vs_duration.png", category: "Diagrams" },
  { label: "SNR vs Depth", src: "/figures/snr_vs_depth.png", category: "Analysis" },
  { label: "T<sub>eff</sub> vs Radius", src: "/figures/teff_vs_radius.png", category: "Analysis" },
  { label: "Size Distribution", src: "/figures/size_distribution_bar.png", category: "Statistics" },
  { label: "Parameter Correlations", src: "/figures/parameter_correlations.png", category: "Statistics" },
  { label: "Quality Scores", src: "/figures/quality_scores.png", category: "Statistics" },
  { label: "Optimization Summary", src: "/figures/optimization_summary.png", category: "Analysis" },
  { label: "Atmosphere Classes", src: "/figures/atmosphere_classification.png", category: "Classification" },
  { label: "Habitability", src: "/figures/habitability_diagram.png", category: "Diagrams" },
  { label: "All Transits", src: "/figures/all_transits_comparison.png", category: "Transits" },
]

const categories = Array.from(new Set(figures.map((f) => f.category)))

export default function AnalysisPage() {
  return (
    <div className="space-y-12 max-w-6xl">
      <div className="space-y-2">
        <h1 className="text-3xl font-light tracking-tight">Analysis</h1>
        <p className="text-sm" style={{ color: 'var(--color-muted)' }}>Scientific visualizations and diagnostic plots</p>
      </div>

      {categories.map((cat) => {
        const figs = figures.filter((f) => f.category === cat)
        return (
          <div key={cat} className="space-y-4">
            <h2 className="text-sm font-medium">{cat}</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {figs.map((fig) => (
                <div key={fig.label} className="space-y-2">
                  <p className="text-xs" style={{ color: 'var(--color-muted)' }} dangerouslySetInnerHTML={{ __html: fig.label }} />
                  <div className="rounded-lg overflow-hidden" style={{ background: 'var(--color-card)', border: '1px solid var(--color-border)' }}>
                    <img
                      src={fig.src}
                      alt={fig.label}
                      className="w-full aspect-[4/3] object-contain"
                      loading="lazy"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none"
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
