import React, { useEffect, useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
// Import Leaflet components (Error handling if not installed)
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
import { Filter, AlertCircle, AlertTriangle, Cpu, ChevronDown, Check, X } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// Fix Leaflet's default icon issue in React
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

// Simple SVG Donut Chart Component
const MiniDonut = ({ label, active, total, colorClass }) => {
  const percentage = total > 0 ? (active / total) * 100 : 0;
  const issues = total - active;
  const radius = 45; 
  const circumference = 2 * Math.PI * radius;
  const activeOffset = circumference - (percentage / 100) * circumference;
  
  return (
    <div className="flex flex-col items-center group relative">
      <div className="relative w-32 h-32"> 
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="64" cy="64" r={radius} 
            stroke="currentColor" strokeWidth="12" fill="transparent" 
            className="text-red-100 hover:opacity-60 cursor-pointer" 
          >
            <title>{`Issues: ${issues}`}</title>
          </circle>
          <circle
            cx="64" cy="64" r={radius} 
            stroke="currentColor" strokeWidth="12" fill="transparent" 
            strokeDasharray={circumference}
            strokeDashoffset={activeOffset}
            strokeLinecap="butt"
            className={`${colorClass} hover:opacity-60 cursor-pointer`} 
          >
             <title>{`Active: ${active}`}</title>
          </circle>
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
           <span className="text-sm font-bold text-teal-600 uppercase tracking-wide">{label}</span>
           <span className="text-lg font-bold text-teal-800 mt-0.5">{active}/{total}</span>
        </div>
      </div>
    </div>
  );
};

const CommandCenter = () => {
  const [stats, setStats] = useState(null);
  const [networkData, setNetworkData] = useState(null);
  const navigate = useNavigate();

  // Filter States
  const [availableTypes, setAvailableTypes] = useState([]);
  const [selectedFilters, setSelectedFilters] = useState([]);
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const filterRef = useRef(null);

  // Helper to normalize issue types (e.g. group "Offline (~2h)" and "Offline (~5m)" as "Offline")
  const getIssueCategory = (issue) => {
    if (issue.startsWith("Offline")) return "Offline";
    return issue;
  };

  useEffect(() => {
    fetch('http://localhost:5000/api/command-center')
      .then(res => res.json())
      .then(data => {
        setStats(data);
        // Extract unique issue types using normalized categories
        if (data.priority_issues) {
          const types = Array.from(new Set(data.priority_issues.map(i => getIssueCategory(i.issue))));
          setAvailableTypes(types);
          
          // Initial load: select all if empty (or update if new types appear)
          setSelectedFilters(prev => {
             if (prev.length === 0) return types;
             // Add any new types that appeared but keep existing selection state
             const newTypes = types.filter(t => !prev.includes(t));
             // If we have new types, we could add them, but standard behavior is usually to respect user selection.
             // For now, if user hasn't interacted (empty prev isn't possible if types existed), just use prev.
             // But if this is first load, types is the set.
             return prev; 
          });
        }
      })
      .catch(console.error);

    fetch('http://localhost:5000/api/network-status')
      .then(res => res.json())
      .then(setNetworkData)
      .catch(console.error);
  }, []);

  // Handle clicking outside of dropdown to close it
  useEffect(() => {
    function handleClickOutside(event) {
      if (filterRef.current && !filterRef.current.contains(event.target)) {
        setIsFilterOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [filterRef]);

  // Sync selected filters when available types change (e.g. initial load)
  useEffect(() => {
    if (availableTypes.length > 0 && selectedFilters.length === 0) {
        setSelectedFilters(availableTypes);
    }
  }, [availableTypes]);

  const toggleFilter = (type) => {
    setSelectedFilters(prev => 
      prev.includes(type) 
        ? prev.filter(t => t !== type) 
        : [...prev, type]
    );
  };

  const toggleAll = () => {
    if (selectedFilters.length === availableTypes.length) {
      setSelectedFilters([]);
    } else {
      setSelectedFilters(availableTypes);
    }
  };

  if (!stats) return <div className="p-10 text-teal-600">Loading Dashboard...</div>;

  const lanProbes = networkData?.buildings?.lan || [];
  const wlanProbes = networkData?.buildings?.wlan || [];
  const lanActive = lanProbes.filter(p => p.color !== 'red').length;
  const wlanActive = wlanProbes.filter(p => p.color !== 'red').length;

  // Marker config
  const getCircleOptions = (color) => {
    const colorMap = {
      green: '#10b981', 
      orange: '#f97316', 
      red: '#ef4444',    
    };
    const hex = colorMap[color] || colorMap.green;
    return {
      color: hex,       
      fillColor: hex,   
      fillOpacity: 0.6,
      weight: 2,        
      radius: 6         
    };
  };

  // Filter Logic using normalized categories
  const filteredIssues = stats.priority_issues.filter(issue => {
    const category = getIssueCategory(issue.issue);
    return selectedFilters.includes(category);
  });

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <h1 className="text-2xl font-bold text-teal-900 mb-6">Network Operations</h1>

      <div className="grid grid-cols-12 gap-6">
        {/* Top Cards Row */}
        <div className="col-span-12 md:col-span-4 mb-2">
           <div className="bg-white rounded-xl border border-teal-100 p-4 shadow-sm flex flex-col justify-center h-full">
             <span className="text-teal-500 text-sm font-semibold uppercase">Active Alerts</span>
             <div className="text-5xl font-bold text-red-500 mt-2">{stats.alerts}</div>
             <div className="text-xs text-red-400 mt-1">Requires immediate attention</div>
           </div>
        </div>
           
        <div className="col-span-12 md:col-span-8 mb-2">
           <div className="bg-white rounded-xl border border-teal-100 p-4 shadow-sm flex flex-col justify-between h-full">
             <span className="text-teal-500 text-sm font-semibold uppercase mb-2">Services Available</span>
             <div className="flex items-center justify-around h-full">
                <MiniDonut label="LAN" active={lanActive} total={lanProbes.length} colorClass="text-blue-500" />
                <div className="w-px h-24 bg-teal-50 mx-4"></div>
                <MiniDonut label="WLAN" active={wlanActive} total={wlanProbes.length} colorClass="text-purple-500" />
             </div>
           </div>
        </div>

        {/* Priority Issues with Specific Filters */}
        <div className="col-span-8 bg-white rounded-xl shadow-sm border border-teal-100 p-6 flex flex-col h-[500px]">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold flex items-center gap-2 text-teal-800">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span> Priority Issues
            </h3>
            
            {/* Dropdown Filter */}
            <div className="relative" ref={filterRef}>
              <button 
                onClick={() => setIsFilterOpen(!isFilterOpen)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-sm font-medium transition-all ${
                  isFilterOpen 
                    ? 'border-teal-500 text-teal-700 bg-teal-50' 
                    : 'border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
              >
                <Filter size={14} />
                <span>Filter Issues</span>
                <ChevronDown size={14} className={`transition-transform ${isFilterOpen ? 'rotate-180' : ''}`} />
                {selectedFilters.length < availableTypes.length && (
                  <span className="ml-1 px-1.5 py-0.5 bg-teal-600 text-white text-[10px] rounded-full">
                    {selectedFilters.length}
                  </span>
                )}
              </button>

              {isFilterOpen && (
                <div className="absolute right-0 top-full mt-2 w-64 bg-white rounded-xl shadow-xl border border-teal-100 z-50 overflow-hidden">
                  <div className="p-3 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
                    <span className="text-xs font-bold text-gray-500 uppercase tracking-wider">Error Types</span>
                    <button 
                      onClick={toggleAll}
                      className="text-xs text-teal-600 hover:text-teal-800 font-medium"
                    >
                      {selectedFilters.length === availableTypes.length ? 'Deselect All' : 'Select All'}
                    </button>
                  </div>
                  <div className="max-h-60 overflow-y-auto p-2 space-y-1">
                    {availableTypes.length === 0 ? (
                      <div className="text-center py-4 text-gray-400 text-xs italic">No active error types found</div>
                    ) : (
                      availableTypes.map(type => (
                        <div 
                          key={type} 
                          onClick={() => toggleFilter(type)}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-teal-50 cursor-pointer transition-colors group"
                        >
                          <div className={`w-4 h-4 rounded border flex items-center justify-center transition-colors ${
                            selectedFilters.includes(type) 
                              ? 'bg-teal-600 border-teal-600' 
                              : 'border-gray-300 group-hover:border-teal-400'
                          }`}>
                            {selectedFilters.includes(type) && <Check size={10} className="text-white" />}
                          </div>
                          <span className={`text-sm ${selectedFilters.includes(type) ? 'text-gray-800' : 'text-gray-500'}`}>
                            {type}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-auto pr-2">
            <table className="w-full text-left">
              <thead className="bg-teal-50 text-xs uppercase text-teal-600 sticky top-0 z-10">
                <tr><th className="p-3 bg-teal-50">Location</th><th className="p-3 bg-teal-50">Issue</th><th className="p-3 bg-teal-50">Action</th></tr>
              </thead>
              <tbody className="divide-y divide-teal-50">
                {filteredIssues.length === 0 ? (
                   <tr><td colSpan="3" className="p-8 text-center text-teal-400 italic">
                     {stats.priority_issues.length > 0 
                       ? "No issues match the selected filters." 
                       : (
                         <div className="flex flex-col items-center gap-2">
                           <CheckCircle size={32} className="text-emerald-400" />
                           <span>All Systems Operational</span>
                         </div>
                       )
                     }
                   </td></tr>
                ) : (
                  filteredIssues.map((issue, idx) => {
                    const cleanLocation = issue.location.replace(/ \((LAN|WLAN)\)$/, '');
                    const isAI = issue.issue.includes("AI") || issue.issue.includes("Anomaly");
                    
                    return (
                      <tr key={idx} className="hover:bg-teal-50/50 transition-colors">
                        <td className="p-3 font-medium text-teal-900">{issue.location}</td>
                        <td className="p-3">
                          <span className={`px-2 py-1 rounded text-xs font-bold inline-flex items-center gap-1.5 ${
                            isAI ? 'bg-purple-100 text-purple-700' :
                            issue.severity === 'High' ? 'bg-red-100 text-red-700' : 
                            'bg-orange-100 text-orange-700'
                          }`}>
                            {isAI && <Cpu size={10} />}
                            {issue.issue}
                          </span>
                        </td>
                        <td className="p-3">
                          <Link to={`/inspector/${cleanLocation}`} className="text-emerald-600 text-sm font-bold hover:underline hover:text-emerald-800">
                            Inspect
                          </Link>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
        
        {/* GEO MAP PANEL */}
        <div className="col-span-4 bg-white rounded-xl border border-teal-100 overflow-hidden shadow-sm flex flex-col h-[500px]">
           <div className="p-4 border-b border-teal-50 flex justify-between items-center">
             <h3 className="font-bold text-teal-800">Probe Locations</h3>
             <span className="text-xs text-teal-500">{stats.map_markers?.length || 0} Points</span>
           </div>
           
           <div className="flex-1 relative z-0">
             {stats.map_markers && stats.map_markers.length > 0 ? (
               <MapContainer 
                 center={[stats.map_markers[0].lat, stats.map_markers[0].lng]} 
                 zoom={15} 
                 style={{ height: "100%", width: "100%" }}
                 scrollWheelZoom={false}
               >
                 <TileLayer
                   attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                   url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                 />
                 {stats.map_markers.map((marker, idx) => (
                   <CircleMarker 
                     key={idx} 
                     center={[marker.lat, marker.lng]} 
                     pathOptions={getCircleOptions(marker.color)}
                     eventHandlers={{
                        click: () => navigate(`/inspector/${marker.name}`)
                     }}
                   >
                     <Tooltip 
                       direction="top" 
                       offset={[0, -10]} 
                       opacity={1}
                       className="bg-transparent border-0 shadow-none"
                     >
                       <div className="text-center bg-white p-2 rounded border border-teal-100 shadow-lg">
                         <strong className="block text-teal-800">{marker.name}</strong>
                         <span className={`text-xs font-bold ${marker.color === 'red' ? 'text-red-600' : 'text-green-600'}`}>
                           {marker.status}
                         </span>
                         <div className="text-[10px] text-gray-400 mt-1 italic">Click to Inspect</div>
                       </div>
                     </Tooltip>
                   </CircleMarker>
                 ))}
               </MapContainer>
             ) : (
               <div className="h-full flex items-center justify-center text-teal-400 bg-teal-50">
                 No GPS data available in GENERAL_info
               </div>
             )}
           </div>
        </div>
      </div>
    </div>
  );
};

export default CommandCenter;