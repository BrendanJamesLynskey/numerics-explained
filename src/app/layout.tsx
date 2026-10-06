/**
 * Root layout for the App Router.
 *
 * Server Component, copied from transformer-explainer's layout (via the companion sites): HTML
 * scaffold, the global stylesheet and the site header. Dark mode follows
 * the system setting (`darkMode: "media"` in tailwind.config.ts).
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/ui/SiteHeader";
import { SITE_URL } from "@/lib/site";

import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "Numerics Explained",
    template: "%s · Numerics Explained",
  },
  description:
    "Number formats, rounding and quantisation for machine learning: FP8, MX and 4-bit formats, stochastic rounding, accumulation error and quantisers, each animated by a bit-exact numerics library.",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
