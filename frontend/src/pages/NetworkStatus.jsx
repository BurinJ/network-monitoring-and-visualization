import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, ArrowDown, ArrowUp, Activity, Filter } from 'lucide-react';

const NetworkStatus = ({ user }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [departments, setDepartments] = useState([]);
  const [selectedDept, setSelectedDept] = useState('All');
  const navigate = useNavigate();

  useEffect(() => {
    fetch('http://localhost:5000/api/network-status')
      .then(res => res.json())
      .then(res => { 
        setData(res); 
        setLoading(false); 
        
        // Extract unique departments
        const allProbes = [...(res.buildings?.lan || []), ...(res.buildings?.wlan || [])];
        const depts = Array.from(new Set(allProbes.map(p => p.department || 'Undefined')));
        setDepartments(['All', ...depts.sort()]);
      })
      .catch(err => setLoading(false));
  }, []);

  const handleInspect = (probeName) => {
    if (user?.role === 'admin') {
      navigate(`/inspector/${probeName}`);
    } else {
      navigate('/login');
    }
  };

  if (loading) return <div className="p-10 text-center text-teal-600 font-medium animate-pulse">Scanning Network...</div>;

  let buildings = data?.buildings || { lan: [], wlan: [] };

  // Filter by Department
  if (selectedDept !== 'All') {
    buildings = {
      lan: buildings.lan.filter(p => (p.department || 'Undefined') === selectedDept),
      wlan: buildings.wlan.filter(p => (p.department || 'Undefined') === selectedDept)
    };
  }

  // Sort Alphabetically by Name
  buildings.lan.sort((a, b) => a.name.localeCompare(b.name));
  buildings.wlan.sort((a, b) => a.name.localeCompare(b.name));

  const StatusCard = ({ item }) => {
    const isClickable = user?.role === 'admin';

    return (
      <div 
        onClick={() => handleInspect(item.name)}
        className={`p-4 border border-teal-100 rounded-xl bg-white group relative transition-all duration-200
          ${isClickable ? 'hover:shadow-md cursor-pointer hover:border-teal-300' : 'opacity-90 cursor-default'}
        `}
      >
        <div className="flex justify-between items-start mb-3">
          <div>
            <span className={`font-bold block text-lg ${isClickable ? 'text-slate-700 group-hover:text-teal-700' : 'text-slate-600'}`}>
              {item.name}
            </span>
            <span className="text-xs text-gray-400 font-medium">{item.department || 'Undefined'} • {item.type}</span>
          </div>
          
          <div className="flex flex-col items-end gap-1">
             {!isClickable && <Lock size={14} className="text-slate-300 mb-1" />}
             <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider
                ${item.color === 'green' ? 'bg-emerald-100 text-emerald-700' : 
                  item.color === 'orange' ? 'bg-orange-100 text-orange-700' : 'bg-red-100 text-red-700'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${item.color === 'green' ? 'bg-emerald-500' : item.color === 'orange' ? 'bg-orange-500' : 'bg-red-500'}`}></span>
                {item.status}
             </span>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-3 gap-2 pt-3 border-t border-teal-50">
           <div className="flex flex-col items-center">
             <div className="flex items-center gap-1 text-xs text-slate-400 mb-0.5">
               <ArrowDown size={12} className="text-emerald-500" /> Down
             </div>
             <span className="font-mono text-sm font-bold text-slate-700">
               {item.down ? item.down : '-'} <span className="text-[10px] font-normal text-slate-400">Mbps</span>
             </span>
           </div>

           <div className="flex flex-col items-center border-l border-teal-50">
             <div className="flex items-center gap-1 text-xs text-slate-400 mb-0.5">
               <ArrowUp size={12} className="text-blue-500" /> Up
             </div>
             <span className="font-mono text-sm font-bold text-slate-700">
               {item.up ? item.up : '-'} <span className="text-[10px] font-normal text-slate-400">Mbps</span>
             </span>
           </div>

           <div className="flex flex-col items-center border-l border-teal-50">
             <div className="flex items-center gap-1 text-xs text-slate-400 mb-0.5">
               <Activity size={12} className="text-purple-500" /> Ping
             </div>
             <span className="font-mono text-sm font-bold text-slate-700">
               {item.latency} <span className="text-[10px] font-normal text-slate-400">ms</span>
             </span>
           </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-8 bg-teal-50 min-h-screen">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-teal-900 mb-2">Connectivity Status</h1>
        <p className="text-teal-600">Real-time performance metrics</p>
        
        {!user && (
          <button 
            onClick={() => navigate('/login')}
            className="mt-4 text-sm text-teal-500 underline hover:text-teal-700"
          >
            Admin Login
          </button>
        )}
      </div>

      {/* Department Filter */}
      <div className="max-w-6xl mx-auto mb-6 flex justify-end">
        <div className="relative inline-block">
          <select 
            value={selectedDept} 
            onChange={(e) => setSelectedDept(e.target.value)}
            className="appearance-none bg-white border border-teal-200 text-teal-800 py-2 pl-4 pr-10 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer font-medium text-sm"
          >
            {departments.map(dept => (
              <option key={dept} value={dept}>{dept}</option>
            ))}
          </select>
          <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-teal-600">
            <Filter size={16} />
          </div>
        </div>
      </div>

      <div className={`max-w-6xl mx-auto border-2 rounded-2xl p-6 flex items-center justify-between mb-12 shadow-sm bg-white
        ${data?.overall === 'Operational' ? 'border-emerald-400' : 'border-red-400'}`}>
        <div>
          <h2 className={`text-2xl font-bold ${data?.overall === 'Operational' ? 'text-emerald-700' : 'text-red-700'}`}>
             {data?.overall === 'Operational' ? '✅ System Healthy' : '⚠️ Service Disruption'}
          </h2>
          <p className="mt-1 text-sm text-slate-600">
            {data?.overall === 'Operational' 
              ? 'All interfaces are performing within optimal parameters.' 
              : 'Issues detected in one or more network segments.'}
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* LAN SECTION */}
        <div>
          <h3 className="text-lg font-bold text-teal-800 mb-4 flex items-center gap-2">
            <span className="p-1.5 bg-teal-100 rounded-lg text-teal-600">🌐</span> Ethernet (LAN)
          </h3>
          <div className="grid grid-cols-1 gap-4">
            {buildings.lan.length === 0 && <div className="text-gray-400 italic text-center py-4">No active LAN monitors found.</div>}
            {buildings.lan.map((place) => (
              <StatusCard key={place.id + 'lan'} item={place} />
            ))}
          </div>
        </div>

        {/* WLAN SECTION */}
        <div>
          <h3 className="text-lg font-bold text-teal-800 mb-4 flex items-center gap-2">
            <span className="p-1.5 bg-purple-100 rounded-lg text-purple-600">📶</span> Wi-Fi (WLAN)
          </h3>
          <div className="grid grid-cols-1 gap-4">
            {buildings.wlan.length === 0 && <div className="text-gray-400 italic text-center py-4">No active Wi-Fi monitors found.</div>}
            {buildings.wlan.map((place) => (
              <StatusCard key={place.id + 'wlan'} item={place} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default NetworkStatus;