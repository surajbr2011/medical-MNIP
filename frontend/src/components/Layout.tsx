import { Outlet, NavLink } from 'react-router-dom';
import { Activity, Search, BarChart2 } from 'lucide-react';

function Layout() {
  return (
    <div className="flex h-screen bg-[#0f111a] text-[#e6edf3] font-sans">
      {/* Sidebar */}
      <aside className="w-64 bg-[#161b22] border-r border-[#30363d] flex flex-col">
        <div className="p-6">
          <h1 className="text-xl font-bold flex items-center gap-2">
            <span>⚕️</span> MNIP Platform
          </h1>
          <p className="text-sm text-gray-400 mt-1">Medical Negligence Intelligence</p>
        </div>
        
        <div className="px-6 mb-4">
          <div className="h-px bg-[#30363d] w-full"></div>
        </div>

        <nav className="flex-1 px-4 space-y-2">
          <p className="px-2 text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Navigate
          </p>
          
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                isActive ? 'bg-[#1f6feb] text-white' : 'hover:bg-[#30363d] text-gray-300'
              }`
            }
          >
            <Activity size={18} />
            Page 1 - Analyse Incident
          </NavLink>
          
          <NavLink
            to="/legal"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                isActive ? 'bg-[#1f6feb] text-white' : 'hover:bg-[#30363d] text-gray-300'
              }`
            }
          >
            <Search size={18} />
            Page 2 - Legal Search
          </NavLink>
          
          <NavLink
            to="/analytics"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg transition-colors ${
                isActive ? 'bg-[#1f6feb] text-white' : 'hover:bg-[#30363d] text-gray-300'
              }`
            }
          >
            <BarChart2 size={18} />
            Page 3 - Analytics Dashboard
          </NavLink>
        </nav>

        <div className="px-6 mt-auto mb-4">
          <div className="h-px bg-[#30363d] w-full mb-4"></div>
          <div className="bg-[#1f242c] p-4 rounded-lg border border-[#30363d]">
            <p className="text-sm text-blue-400 font-medium">ℹ️ ABDM FHIR R4 Compliant</p>
            <p className="text-xs text-gray-400 mt-1">Medico-Legal Intelligence</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <div className="p-8 max-w-7xl mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export default Layout;
