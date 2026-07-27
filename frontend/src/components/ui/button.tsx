"use client"
import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost" | "danger"
  size?: "sm" | "md" | "lg"
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center rounded-lg font-medium transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:ring-offset-background disabled:pointer-events-none disabled:opacity-50",
          variant === "primary" && "bg-primary text-primary-foreground hover:bg-primary/90 active:scale-[0.98]",
          variant === "secondary" && "bg-secondary text-secondary-foreground hover:bg-secondary/80 border border-border active:scale-[0.98]",
          variant === "ghost" && "text-muted-foreground hover:bg-secondary hover:text-foreground",
          variant === "danger" && "bg-destructive text-destructive-foreground hover:bg-destructive/90",
          size === "sm" && "h-8 px-3 text-xs gap-1.5",
          size === "md" && "h-10 px-4 text-sm gap-2",
          size === "lg" && "h-12 px-6 text-base gap-2.5",
          className
        )}
        {...props}
      >
        {children}
      </button>
    )
  }
)
Button.displayName = "Button"
