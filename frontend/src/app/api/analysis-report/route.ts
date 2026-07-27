import { NextResponse } from "next/server"
import { readFile } from "fs/promises"
import { join } from "path"

export async function GET() {
  try {
    const filePath = join(process.cwd(), "..", "outputs", "reports", "analysis_report.json")
    const data = await readFile(filePath, "utf-8")
    return NextResponse.json(JSON.parse(data))
  } catch {
    return NextResponse.json({ error: "Failed to load analysis report" }, { status: 500 })
  }
}
