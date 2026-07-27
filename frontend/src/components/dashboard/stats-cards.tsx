"use client"
import { motion } from "framer-motion"
import { Card, CardTitle, CardValue } from "@/components/ui/card"
import { Telescope, CheckCircle2, Activity, Thermometer, BarChart3, Cpu } from "lucide-react"
import { type SummaryStats } from "@/lib/data"

interface StatsCardsProps {
  stats: SummaryStats | null
}

const statConfigs = [
  { key: "n_candidates", label: "Candidates", icon: Telescope, suffix: "", decimals: 0, color: "text-primary" },
  { key: "n_promising", label: "Promising", icon: Activity, suffix: "", decimals: 0, color: "text-chart-3" },
  { key: "n_confirmed", label: "Confirmed", icon: CheckCircle2, suffix: "", decimals: 0, color: "text-success" },
  { key: "median_period", label: "Median Period", icon: BarChart3, suffix: " days", decimals: 2, color: "text-chart-2" },
  { key: "median_radius", label: "Median Radius", icon: Cpu, suffix: " R⊕", decimals: 3, color: "text-chart-4" },
  { key: "median_teq", label: "Median T<sub>eq</sub>", icon: Thermometer, suffix: " K", decimals: 0, color: "text-warning" },
]

export function StatsCards({ stats }: StatsCardsProps) {
  if (!stats) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <Card key={i} className="h-24 animate-skeleton-pulse" />
        ))}
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
      {statConfigs.map((cfg, i) => {
        const value = stats[cfg.key as keyof SummaryStats] as number
        const Icon = cfg.icon
        return (
          <Card key={cfg.key} index={i} hover>
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <CardTitle>{cfg.label}</CardTitle>
                <CardValue className={cfg.color}>
                  {typeof value === "number"
                    ? value.toFixed(cfg.decimals) + cfg.suffix
                    : "—"}
                </CardValue>
              </div>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-secondary">
                <Icon className="h-4 w-4 text-muted-foreground" />
              </div>
            </div>
          </Card>
        )
      })}
    </div>
  )
}
