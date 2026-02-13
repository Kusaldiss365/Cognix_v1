import type { Metadata } from "next";
import { Geist, Geist_Mono, Orbitron } from "next/font/google";
import "./globals.css";
import Header from "@/component/Header";
import { Toaster } from 'sonner';


const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const orbitron = Orbitron({
  variable: "--font-orbitron",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});


export const metadata: Metadata = {
  title: "CogniX",
  description: "AI studdy buddy",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
return (
  <html lang="en">
    <body
      className={`${geistSans.variable} ${geistMono.variable} ${orbitron.variable} font-[var(--font-orbitron)] antialiased min-h-screen flex flex-col overflow-hidden`}
      style={{
        backgroundImage: "url('/bg_network.png')",
        backgroundRepeat: "no-repeat",
        backgroundSize: "cover",
        backgroundPosition: "center",
        margin: 0,
        padding: 0,
      }}
    >
      <Header />
      <Toaster richColors position="bottom-right" />
      <main className="flex-1 flex items-center justify-center w-full">
        {children}
      </main>
    </body>
  </html>
);

}

