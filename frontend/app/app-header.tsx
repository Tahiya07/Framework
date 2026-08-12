"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const links = [
  { href: "/", label: "Home" },
  { href: "/student", label: "Student" },
  { href: "/teacher", label: "Teacher" },
  { href: "/settings", label: "System status" },
];

export default function AppHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const active = (href: string) => href === "/" ? pathname === href : pathname.startsWith(href);

  return (
    <header className="sticky top-0 z-40 mx-auto w-[min(1260px,100%)] px-3 pt-3 sm:px-7">
      <nav aria-label="Primary navigation" className="glass-card-light relative flex min-h-16 items-center justify-between rounded-[22px] px-4 sm:px-6">
        <Link href="/" onClick={() => setOpen(false)} className="display flex shrink-0 items-center gap-2 text-lg font-extrabold tracking-[-.06em] text-[#16213d]">
          <span className="grid h-7 w-7 place-items-center rounded-full bg-[#355cda] text-base text-white">F</span>
          Framework
        </Link>
        <div className="desktop-nav flex items-center gap-2" aria-label="Desktop navigation">
          {links.map((link) => <Link key={link.href} className={`nav-link rounded-full px-3 py-2 ${active(link.href) ? "bg-white/55 text-[#294f7b] shadow-sm" : ""}`} href={link.href}>{link.label}</Link>)}
        </div>
        <button className="mobile-menu-button btn-secondary !rounded-full !px-3 !py-2 text-xs" aria-expanded={open} aria-controls="mobile-navigation" aria-label="Toggle navigation menu" onClick={() => setOpen(value => !value)}>
          <span aria-hidden="true">{open ? "×" : "☰"}</span><span className="sr-only">Menu</span>
        </button>
        <div id="mobile-navigation" className={`mobile-nav glass-card absolute left-0 right-0 top-[calc(100%+.55rem)] rounded-[22px] p-2 ${open ? "mobile-nav-open" : ""}`}>
          {links.map((link) => <Link key={link.href} href={link.href} onClick={() => setOpen(false)} className={`nav-link block rounded-xl px-4 py-3 ${active(link.href) ? "bg-white/65 text-[#294f7b]" : ""}`}>{link.label}</Link>)}
        </div>
      </nav>
    </header>
  );
}
