"use client"
import { usePathname } from "next/navigation"
import { Menu, Search, Bell } from "lucide-react"
import { cn } from "@/lib/utils"

const pageTitles: Record<string, string> = {
  "/": "Dashboard",
  "/candidates": "Candidates",
  "/classification": "Classification",
  "/analysis": "Analysis",
}

interface NavbarProps {
  onMenuClick: () => void
}

export function Navbar({ onMenuClick }: NavbarProps) {
  const pathname = usePathname()
  const title = Object.entries(pageTitles).find(([key]) =>
    key === "/" ? pathname === "/" : pathname.startsWith(key)
  )?.[1] || "ExoVista"

  const isDetail = pathname.startsWith("/candidates/") && pathname !== "/candidates"

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-4 border-b border-border bg-background/80 backdrop-blur-xl px-4 lg:px-6">
      <button
        onClick={onMenuClick}
        className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors lg:hidden"
      >
        <Menu className="h-4 w-4" />
      </button>

      <div className="flex-1">
        <h1 className="text-sm font-semibold">
          {title}
          {isDetail && (
            <span className="ml-2 text-muted-foreground font-normal">
              / {pathname.split("/").pop()}
            </span>
          )}
        </h1>
      </div>

      <div className="flex items-center gap-2">
        <button className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
          <Search className="h-4 w-4" />
        </button>
        <button className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors">
          <Bell className="h-4 w-4" />
        </button>
        <div className="ml-2 flex h-7 w-7 items-center justify-center rounded-full bg-primary/20 text-primary text-xs font-semibold">
          ED
        </div>
      </div>
    </header>
  )
}
