import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  TrendingUp, 
  ChevronLeft, 
  ChevronRight, 
  AlertTriangle, 
  Search,
  LayoutDashboard 
} from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { API_BASE_URL } from '../config'; 

const Sidebar = ({ isOpen, toggleSidebar }) => {
  const location = useLocation();
  const [defaultProbe, setDefaultProbe] = useState('Library');

  // Fetch probes only to determine where the main "Probe Inspection" link should go
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/probes`)
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) {
          setDefaultProbe(data[0]); // Default to the first probe in the list
        }
      })
      .catch(err => console.error("Failed to fetch probes:", err));
  }, []);

  const NavItem = ({ to, icon, label }) => {
    // Highlight if the path starts with the link (e.g., /inspector highlights for /inspector/Library)
    const isActive = location.pathname === to || (to !== '/' && location.pathname.startsWith(to));
    return (
      <Link 
        to={to} 
        className={`flex items-center gap-3 p-3 rounded-lg mb-2 transition-colors duration-200
          ${isActive 
            ? 'bg-emerald-600 text-white shadow-md' 
            : 'text-teal-100 hover:bg-teal-800 hover:text-white'}
          ${!isOpen && 'justify-center'}
        `}
        title={!isOpen ? label : ''}
      >
        <div className="min-w-[20px]">{icon}</div>
        
        <span className={`font-medium transition-all duration-200 overflow-hidden whitespace-nowrap ${
          isOpen ? 'w-auto opacity-100' : 'w-0 opacity-0 hidden'
        }`}>
          {label}
        </span>
      </Link>
    );
  };

  return (
    <div 
      className={`bg-teal-900 h-screen fixed left-0 top-0 flex flex-col text-white shadow-xl transition-all duration-300 ease-in-out z-50
        ${isOpen ? 'w-64' : 'w-20'}
      `}
    >
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-teal-800 relative">
        <div className={`font-bold text-[20px] tracking-wider text-emerald-400 transition-opacity duration-200 ${isOpen ? 'opacity-100' : 'opacity-0 hidden'}`}>
          KU Net
        </div>
        
        <button 
          onClick={toggleSidebar}
          className="p-1 rounded text-teal-300 hover:text-white hover:bg-teal-800 transition-colors absolute right-4"
          style={!isOpen ? { right: '50%', transform: 'translateX(50%)' } : {}}
        >
          {isOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-6 overflow-y-auto scrollbar-hide">
        <NavItem to="/" icon={<Activity size={20} />} label="Connectivity Status" />
        
        {/* Renamed to "Network Operations" */}
        <NavItem to="/admin" icon={<LayoutDashboard size={20} />} label="Monitoring Dashboard" />
        
        <NavItem to={`/inspector/${defaultProbe}`} icon={<Search size={20} />} label="Network Inspection" />
        
        <NavItem to="/trends" icon={<TrendingUp size={20} />} label="Trends" />
      </nav>

      {/* Profile */}
      <div className="p-4 border-t border-teal-800 overflow-hidden bg-teal-950/30 shrink-0">
        <div className={`flex items-center gap-3 transition-all duration-300 ${!isOpen && 'justify-center'}`}>
          <div className="w-8 h-8 min-w-[32px] rounded-full bg-emerald-500 flex items-center justify-center font-bold text-white shadow-sm">
            A
          </div>
          <div className={`text-sm transition-opacity duration-200 ${isOpen ? 'opacity-100' : 'opacity-0 hidden'}`}>
            <div className="font-semibold whitespace-nowrap text-teal-50">Admin</div>
            <div className="text-xs text-teal-400 whitespace-nowrap">IT Department</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;