import React from 'react';
import { LayoutDashboard, FileText, Activity, ChevronLeft, ChevronRight } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

// Receive props from App.jsx
const Sidebar = ({ isOpen, toggleSidebar }) => {
  const location = useLocation();

  const NavItem = ({ to, icon, label }) => {
    const isActive = location.pathname === to;
    return (
      <Link 
        to={to} 
        className={`flex items-center gap-3 p-3 rounded-lg mb-2 transition-colors duration-200
          ${isActive ? 'bg-[#29ff69] text-black' : 'text-white hover:bg-[#00cf6e] hover:text-gray-600'}
          ${!isOpen && 'justify-center'} /* Center icons when closed */
        `}
        title={!isOpen ? label : ''} // Show tooltip on hover when closed
      >
        {/* Force icon size to stay consistent */}
        <div className="min-w-[20px]">{icon}</div>
        
        {/* Hide text when closed */}
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
      className={`bg-[#02BC77] h-screen fixed left-0 top-0 flex flex-col text-white shadow-xl transition-all duration-300 ease-in-out
        ${isOpen ? 'w-64' : 'w-20'} /* Dynamic Width */
      `}
    >
      {/* Header / Logo Area */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-slate-800 relative">
        
        {/* Logo Text - Hides when closed */}
        <div className={`font-bold text-lg tracking-wider transition-opacity duration-200 ${isOpen ? 'opacity-100' : 'opacity-0 hidden'}`}>
          KU Network
        </div>
        
        {/* Toggle Button */}
        <button 
          onClick={toggleSidebar}
          className="p-0 rounded text-gray-200 hover:text-white transition-colors absolute right-4"
          // If closed, center the button:
          style={!isOpen ? { right: '50%', transform: 'translateX(50%)' } : {}}
        >
          {isOpen ? <ChevronLeft size={30} /> : <ChevronRight size={30} />}
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-6">
        <NavItem to="/" icon={<LayoutDashboard size={20} />} label="Overview" />
        <NavItem to="/report" icon={<FileText size={20} />} label="Reports" />
        <NavItem to="/probe/1" icon={<Activity size={20} />} label="Probe Details" />
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-slate-800 overflow-hidden">
        <div className={`flex items-center gap-3 transition-all duration-300 ${!isOpen && 'justify-center'}`}>
          <div className="w-8 h-8 min-w-[32px] rounded-full bg-blue-500 flex items-center justify-center font-bold">
            A
          </div>
          
          {/* Text Area */}
          <div className={`text-sm transition-opacity duration-200 ${isOpen ? 'opacity-100' : 'opacity-0 hidden'}`}>
            <div className="font-semibold whitespace-nowrap">Admin</div>
            <div className="text-xs text-gray-500 whitespace-nowrap">IT Department</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;