"use client";

/**
 * Code-split client widgets: each loads its own chunk after the page shell,
 * so pages stay light (the pattern of the companion sites' lazy.tsx). Each
 * widget takes its equation as server-rendered children.
 */
import dynamic from "next/dynamic";

function Placeholder({ what }: { what: string }): JSX.Element {
  return (
    <p
      data-pending-widget
      className="my-8 min-h-96 text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the {what}…
    </p>
  );
}

const loading = (what: string) =>
  function Loading(): JSX.Element {
    return <Placeholder what={what} />;
  };

export const BitsWidget = dynamic(() => import("./BitsWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const RoundingWidget = dynamic(() => import("./RoundingWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const StagnationWidget = dynamic(() => import("./StagnationWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const SumWidget = dynamic(() => import("./SumWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const ProbeWidget = dynamic(() => import("./ProbeWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const MxWidget = dynamic(() => import("./MxWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const GranularityWidget = dynamic(() => import("./GranularityWidget"), {
  ssr: false,
  loading: loading("animation"),
});
export const ZeroPointWidget = dynamic(() => import("./ZeroPointWidget"), {
  ssr: false,
  loading: loading("animation"),
});
