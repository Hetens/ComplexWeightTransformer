"use client";

import { useEffect, useRef, useState } from "react";

/**
 * True once the element has been on screen. Panels use it to gate their forward passes.
 *
 * Without this every panel queues work on the single worker the moment the page mounts: four model
 * downloads and a dozen probe passes, all ahead of whatever the reader is actually looking at. It
 * never unsets, so scrolling back to a panel does not recompute what it already has.
 */
export function useInView<T extends HTMLElement>(rootMargin = "200px") {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    // No IntersectionObserver: show everything. Deferred to a microtask rather than set inline,
    // because the initial value cannot depend on a browser API the server does not have.
    if (typeof IntersectionObserver === "undefined") { queueMicrotask(() => setSeen(true)); return; }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setSeen(true); io.disconnect(); } },
      { rootMargin });
    io.observe(el);
    return () => io.disconnect();
  }, [seen, rootMargin]);
  return [ref, seen] as const;
}
