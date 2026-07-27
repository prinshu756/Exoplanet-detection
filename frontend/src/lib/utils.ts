import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(num: number, decimals = 2): string {
  if (Number.isNaN(num) || num === undefined || num === null) return "—"
  if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + "M"
  if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + "K"
  return num.toFixed(decimals)
}

export function formatScientific(num: number): string {
  if (Number.isNaN(num) || num === undefined || num === null) return "—"
  if (num === 0) return "0"
  if (Math.abs(num) < 0.01 || Math.abs(num) >= 10000) {
    return num.toExponential(2)
  }
  return num.toFixed(4)
}

export function slugify(text: string): string {
  return text.toLowerCase().replace(/\s+/g, "-")
}
