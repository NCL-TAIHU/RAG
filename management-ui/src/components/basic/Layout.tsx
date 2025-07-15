import { useState } from 'react'
import Sidebar from './Sidebar'
import type { ReactNode } from 'react'
import './Layout.css'


export default function Layout({ children }: { children: ReactNode }) {
  const [isSidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="layout">
      <button className="hamburger" onClick={() => setSidebarOpen(!isSidebarOpen)}>
        ☰
      </button>
      <div className={`sidebar-container ${isSidebarOpen ? 'open' : ''}`}>
        <Sidebar onNavigate={() => setSidebarOpen(false)} />
      </div>
      <main className="main">{children}</main>
    </div>
  )
}
