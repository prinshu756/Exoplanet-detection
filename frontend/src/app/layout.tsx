"use client"
import { usePathname } from "next/navigation"
import { AnimatePresence, motion } from "framer-motion"
import Link from "next/link"
import "./globals.css"

const navItems = [
  { label: "Dashboard", href: "/" },
  { label: "Candidates", href: "/candidates" },
  { label: "Classification", href: "/classification" },
  { label: "Analysis", href: "/analysis" },
]

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()

  return (
    <html lang="en">
      <head>
        <title>ExoVista — Exoplanet Detection</title>
        <meta name="description" content="AI-powered exoplanet detection platform" />
      </head>
      <body>
        <header className="border-b border-border" style={{ background: 'var(--color-bg)' }}>
          <div className="container-main flex items-center justify-between h-16">
            <Link href="/" className="text-sm font-medium tracking-tight" style={{ color: 'var(--color-fg)' }}>
              ExoVista
            </Link>
            <nav className="flex items-center gap-8">
              {navItems.map((item) => {
                const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="text-sm transition-all duration-200"
                    style={{
                      color: isActive ? 'var(--color-fg)' : 'var(--color-muted)',
                      fontWeight: isActive ? 500 : 400,
                    }}
                  >
                    {item.label}
                  </Link>
                )
              })}
            </nav>
          </div>
        </header>

        <main className="container-main py-12 lg:py-16">
          <AnimatePresence mode="wait">
            <motion.div
              key={pathname}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>

        <footer className="border-t border-border py-8" style={{ background: 'var(--color-bg)' }}>
          <div className="container-main flex items-center justify-between text-xs" style={{ color: 'var(--color-muted)' }}>
            <span>ExoVista © 2026</span>
            <span>AI-powered exoplanet detection</span>
          </div>
        </footer>
      </body>
    </html>
  )
}
