import { NavLink } from 'react-router-dom'
import './Sidebar.css'
interface SidebarProps {
    onNavigate?: () => void;
}
export default function Sidebar({ onNavigate }: SidebarProps) {
    const handleClick = () => {
      if (onNavigate) onNavigate();
    };
  
    return (
      <aside className="sidebar">
        <h2>TAIHU</h2>
        <nav className="nav">
          <NavLink to="/apps" onClick={handleClick}>Apps</NavLink>
          <NavLink to="/benchmarks" onClick={handleClick}>Benchmarks</NavLink>
        </nav>
      </aside>
    );
}