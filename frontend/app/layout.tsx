import "./globals.css";
import PwaInstall from "./pwa-install";
import AppHeader from "./app-header";
export const metadata = { title: "Framework · Academic intelligence", description: "Private, local Bloom-aware academic assistance", manifest: "/manifest.webmanifest", icons: { icon: "/framework-icon.svg" } };
export const viewport = { width: "device-width", initialScale: 1, viewportFit: "cover", themeColor: "#294f7b" };
export default function Layout({children}:{children:React.ReactNode}) { return <html lang="en"><body><div className="app-shell"><AppHeader/>{children}<PwaInstall/></div></body></html> }
