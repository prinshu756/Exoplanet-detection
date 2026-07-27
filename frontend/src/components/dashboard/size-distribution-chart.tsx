"use client"
import { useMemo } from "react"
import { Card, CardHeader, CardTitle } from "@/components/ui/card"
import { type Candidate, classifySize } from "@/lib/data"

interface SizeDistributionChartProps {
  candidates: Candidate[]
}

const SIZE_ORDER = ["Sub-Earth", "Earth-size", "Super-Earth", "Sub-Neptune", "Neptune-size", "Jupiter-size"]
const SIZE_COLORS: Record<string, string> = {
  "Sub-Earth": "#b0b0b0",
  "Earth-size": "#4caf50",
  "Super-Earth": "#2196f3",
  "Sub-Neptune": "#ff9800",
  "Neptune-size": "#f44336",
  "Jupiter-size": "#9c27b0",
}

export function SizeDistributionChart({ candidates }: SizeDistributionChartProps) {
  const distribution = useMemo(() => {
    const counts: Record<string, number> = {}
    candidates.forEach((c) => {
      const size = classifySize(c.planet_radius_rearth)
      counts[size] = (counts[size] || 0) + 1
    })
    return SIZE_ORDER.map((label) => ({
      label,
      count: counts[label] || 0,
      color: SIZE_COLORS[label],
    }))
  }, [candidates])

  const maxCount = Math.max(...distribution.map((d) => d.count), 1)

  return (
    <Card className="p-5">
      <CardHeader>
        <CardTitle>Size Distribution</CardTitle>
      </CardHeader>
      <div className="space-y-3">
        {distribution.map((item) => (
          <div key={item.label} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">{item.label}</span>
              <span className="font-medium">{item.count}</span>
            </div>
            <div className="h-2 rounded-full bg-secondary overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{
                  width: `${(item.count / maxCount) * 100}%`,
                  backgroundColor: item.color,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </Card>
  )
}
