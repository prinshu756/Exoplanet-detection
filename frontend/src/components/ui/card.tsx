"use client"
import { cn } from "@/lib/utils"
import { motion } from "framer-motion"

interface CardProps {
  className?: string
  children?: React.ReactNode
  hover?: boolean
  index?: number
  onClick?: () => void
}

export function Card({ className, children, hover = false, index = 0, onClick }: CardProps) {
  const Comp = onClick ? "button" : motion.div
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: index * 0.05, ease: [0.16, 1, 0.3, 1] }}
      className={cn(
        "rounded-xl border border-border bg-card p-6",
        hover && "card-hover cursor-pointer",
        className
      )}
      onClick={onClick}
    >
      {children}
    </motion.div>
  )
}

export function CardHeader({ className, children }: { className?: string; children: React.ReactNode }) {
  return <div className={cn("mb-4", className)}>{children}</div>
}

interface CardTitleProps {
  className?: string;
  children?: React.ReactNode;
  dangerouslySetInnerHTML?: { __html: string };
}
export function CardTitle({ className, children, dangerouslySetInnerHTML }: CardTitleProps) {
  if (dangerouslySetInnerHTML) {
    return <h3 className={cn("text-sm font-medium text-muted-foreground tracking-wide uppercase", className)} dangerouslySetInnerHTML={dangerouslySetInnerHTML} />
  }
  return <h3 className={cn("text-sm font-medium text-muted-foreground tracking-wide uppercase", className)}>{children}</h3>
}

export function CardValue({ className, children }: { className?: string; children: React.ReactNode }) {
  return <p className={cn("text-2xl font-semibold tracking-tight mt-1", className)}>{children}</p>
}
