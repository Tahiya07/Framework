"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const links = [
  { href: "/", label: "Home" },
  { href: "/student", label: "Student" },
  { href: "/teacher", label: "Teacher" },
];

export default function AppHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const active = (href: string) => href === "/" ? pathname === href : pathname.startsWith(href);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, []);

  return <>
    <header className="sticky top-0 z-40 mx-auto w-[min(1260px,100%)] px-3 pt-3 sm:px-7">
      <nav aria-label="Primary navigation" className="glass-card-light relative flex min-h-16 items-center justify-between rounded-[22px] px-4 sm:px-6">
        <Link href="/" onClick={() => setOpen(false)} className="display flex shrink-0 items-center gap-2 text-lg font-extrabold tracking-[-.06em] text-[#16213d]"><span className="grid h-7 w-7 place-items-center rounded-full bg-[#355cda] text-base text-white">F</span>Framework</Link>
        <div className="desktop-nav flex items-center gap-2" aria-label="Desktop navigation">{links.map((link) => <Link key={link.href} className={`nav-link rounded-full px-3 py-2 ${active(link.href) ? "bg-white/55 text-[#294f7b] shadow-sm" : ""}`} href={link.href}>{link.label}</Link>)}</div>
        <button className="mobile-menu-button" aria-expanded={open} aria-controls="mobile-navigation" aria-label="Open navigation menu" onClick={() => setOpen(true)}><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round"><path d="M4 7h16M4 12h16M4 17h16" /></svg><span className="sr-only">Menu</span></button>
      </nav>
    </header>
    <div id="mobile-navigation" className={`mobile-nav ${open ? "mobile-nav-open" : ""}`} aria-hidden={!open}>
      <button className="mobile-nav-backdrop" onClick={() => setOpen(false)} aria-label="Close navigation menu" tabIndex={open ? 0 : -1} />
      <aside className="mobile-nav-panel" role="dialog" aria-modal="true" aria-label="Framework navigation">
        <div className="mobile-nav-heading"><div><span className="mobile-nav-kicker">Framework</span><h2>Explore your workspace</h2></div><button onClick={() => setOpen(false)} className="mobile-nav-close" aria-label="Close navigation menu"><svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="m6 6 12 12M18 6 6 18" /></svg></button></div>
        <nav className="mobile-nav-links" aria-label="Mobile navigation">{links.map((link) => <Link key={link.href} href={link.href} onClick={() => setOpen(false)} className={`mobile-nav-link ${active(link.href) ? "mobile-nav-link-active" : ""}`}><span>{link.label}</span><span aria-hidden="true">›</span></Link>)}</nav>
        <p className="mobile-nav-foot">Private, local academic intelligence<br />for focused learning and assessment.</p>
      </aside>
    </div>
  </>;
}
