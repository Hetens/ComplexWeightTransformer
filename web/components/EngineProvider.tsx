"use client";

import { createContext, ReactNode, useContext } from "react";
import { useEngine } from "@/lib/useEngine";

const Ctx = createContext<ReturnType<typeof useEngine> | null>(null);

export function EngineProvider({ children }: { children: ReactNode }) {
  const engine = useEngine();
  return <Ctx.Provider value={engine}>{children}</Ctx.Provider>;
}

export function useEngineCtx() {
  const c = useContext(Ctx);
  if (!c) throw new Error("useEngineCtx outside EngineProvider");
  return c;
}

export const MODEL_IDS = [
  "complex-nope-d160-s0", "real-nope-d224-s0", "real-nope-d160-s0", "real-rope-d160-s0",
] as const;
export type ModelId = (typeof MODEL_IDS)[number];
