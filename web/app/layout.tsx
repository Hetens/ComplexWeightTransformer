import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Phase Without Position — an interactive companion",
  description:
    "Run the trained transformers from the paper in your browser: the reverse-task result, the "
    + "fractional positional code, and the randomization test behind it.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
