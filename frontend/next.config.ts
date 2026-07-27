import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow self-sourced images (local static files)
  images: {
    unoptimized: true,
  },
  // Configure CORS headers for API routes
  async headers() {
    return [
      {
        source: "/api/:path*",
        headers: [
          { key: "Access-Control-Allow-Origin", value: "*" },
        ],
      },
    ];
  },
};

export default nextConfig;
