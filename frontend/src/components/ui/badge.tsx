import { cn } from "@/lib/utils"

interface BadgeProps {
  variant?: "default" | "success" | "warning" | "danger" | "info"
  className?: string
  children: React.ReactNode
}

export function Badge({ variant = "default", className, children }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
        variant === "default" && "bg-secondary text-muted-foreground",
        variant === "success" && "bg-success/15 text-success",
        variant === "warning" && "bg-warning/15 text-warning",
        variant === "danger" && "bg-danger/15 text-danger",
        variant === "info" && "bg-primary/15 text-primary",
        className
      )}
    >
      {children}
    </span>
  )
}
