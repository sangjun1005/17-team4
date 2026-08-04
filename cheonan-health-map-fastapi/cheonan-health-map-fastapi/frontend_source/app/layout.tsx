import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "천안시 지도",
  description: "천안시 범위에 집중한 무료 웹지도 시안",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
