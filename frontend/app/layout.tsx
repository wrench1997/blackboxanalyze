import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("http://127.0.0.1:3000"),
  title: "SIFT Research — Memory-first Rule Learning",
  description: "A research workbench for black-box rule induction, long-term memory and root-cause-driven model improvement.",
  openGraph: {
    title: "SIFT — Black-box Rule Lab",
    description: "Three-seed family holdout research console for executable rule memory and structured browser semantics.",
    images: [{ url: "/og.png", width: 1672, height: 941, alt: "SIFT black-box rule lab experiment card" }],
  },
  twitter: {
    card: "summary_large_image",
    title: "SIFT — Black-box Rule Lab",
    description: "Three seeds, two held-out frontend vulnerability families, 100% structured holdout in the research pilot.",
    images: ["/og.png"],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
