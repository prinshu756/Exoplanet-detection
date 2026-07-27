import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

export async function GET() {
  try {
    const filePath = join(process.cwd(), "..", "outputs", "candidate_parameters", "candidate_parameters.json")
    const data = await readFile(filePath, "utf-8")
    const candidates = JSON.parse(data)
    const cleaned = candidates.map((c: Record<string, unknown>) => {
      const obj: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(c)) {
        obj[key] = (typeof value === "number" && isNaN(value)) ? null : value
      }
      return obj
    })
    return NextResponse.json(cleaned)
  } catch {
    return NextResponse.json({ error: "Failed to load candidates" }, { status: 500 })
  }
}
