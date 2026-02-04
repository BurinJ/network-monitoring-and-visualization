import React, { useEffect, useState, useRef } from 'react';
import { Link, useNavigate } from 'react-router-dom';
// Import Leaflet components (Error handling if not installed)
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
import { Filter, AlertCircle, AlertTriangle, Cpu, ChevronDown, Check, X, Layers, Network, Wifi, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
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
  
  // Department Filter
  const [departments, setDepartments] = useState([]);
  const [selectedDept, setSelectedDept] = useState('All');
  
  // Heatmap Toggle
  const [heatmapMode, setHeatmapMode] = useState('WLAN'); // 'LAN' or 'WLAN'

  // Sort State
  const [sortConfig, setSortConfig] = useState({ key: 'severity', direction: 'desc' });
  
  const filterRef = useRef(null);

  // Helper to normalize issue types
  const getIssueCategory = (issue) => {
    if (issue.startsWith("Offline")) return "Offline";
    return issue;
  };

  useEffect(() => {
    // Fetch Command Center Stats
    fetch('http://localhost:5000/api/command-center')
      .then(res => res.json())
      .then(data => {
        setStats(data);
        if (data.priority_issues) {
          const types = Array.from(new Set(data.priority_issues.map(i => getIssueCategory(i.issue))));
          setAvailableTypes(types);
          setSelectedFilters(prev => {
             if (prev.length === 0) return types;
             const newTypes = types.filter(t => !prev.includes(t));
             return prev; 
          });
        }
      })
      .catch(console.error);

    // Fetch Network Status (Used for donut, filters, and heatmap)
    fetch('http://localhost:5000/api/network-status')
      .then(res => res.json())
      .then(data => {
        setNetworkData(data);
        const allProbes = [...(data.buildings?.lan || []), ...(data.buildings?.wlan || [])];
        const depts = Array.from(new Set(allProbes.map(p => p.department || 'Undefined')));
        setDepartments(['All', ...depts.sort()]);
      })
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

  // Sync selected filters when available types change
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

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    } else if (key === 'severity' && sortConfig.key !== 'severity') {
      direction = 'desc'; // Default to high-to-low for severity
    }
    setSortConfig({ key, direction });
  };

  if (!stats) return <div className="p-10 text-teal-600">Loading Dashboard...</div>;

  // Filter Data by Department
  const lanProbes = (networkData?.buildings?.lan || []).filter(p => selectedDept === 'All' || (p.department || 'Undefined') === selectedDept);
  const wlanProbes = (networkData?.buildings?.wlan || []).filter(p => selectedDept === 'All' || (p.department || 'Undefined') === selectedDept);
  
  const lanActive = lanProbes.filter(p => p.color !== 'red').length;
  const wlanActive = wlanProbes.filter(p => p.color !== 'red').length;

  const visibleProbeNames = new Set([...lanProbes, ...wlanProbes].map(p => p.name));
  const filteredMapMarkers = (stats.map_markers || []).filter(m => visibleProbeNames.has(m.name));

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
      radius: 4         
    };
  };

  const filteredIssues = stats.priority_issues.filter(issue => {
    const category = getIssueCategory(issue.issue);
    const probeName = issue.location.replace(/ \((LAN|WLAN)\)$/, '');
    const probeData = [...(networkData?.buildings?.lan || []), ...(networkData?.buildings?.wlan || [])].find(p => p.name === probeName);
    const probeDept = probeData ? (probeData.department || 'Undefined') : 'Undefined';
    const matchesDept = selectedDept === 'All' || probeDept === selectedDept;
    return matchesDept && selectedFilters.includes(category);
  });

  // Sort Issues
  const sortedIssues = [...filteredIssues].sort((a, b) => {
    if (sortConfig.key === 'location') {
        const valA = a.location.toLowerCase();
        const valB = b.location.toLowerCase();
        if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
        if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
    }
    if (sortConfig.key === 'severity') {
        const weight = { 'High': 3, 'Medium': 2, 'Low': 1 }; 
        const wA = weight[a.severity] || 0;
        const wB = weight[b.severity] || 0;
        
        if (wA !== wB) return sortConfig.direction === 'asc' ? wA - wB : wB - wA;
        return a.location.localeCompare(b.location);
    }
    if (sortConfig.key === 'issue') {
        return sortConfig.direction === 'asc' 
           ? a.issue.localeCompare(b.issue)
           : b.issue.localeCompare(a.issue);
    }
    return 0;
  });

  // Filter Heatmap based on Mode (LAN/WLAN) and Department
  const heatmapSource = networkData?.buildings?.[heatmapMode.toLowerCase()] || [];
  const filteredHeatmap = heatmapSource.filter(h => {
     const probeDept = h.department || 'Undefined';
     return selectedDept === 'All' || probeDept === selectedDept;
  });

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <h1 className="text-2xl font-bold text-teal-900">Network Operations</h1>
        
        <div className="relative inline-block">
            <select 
            value={selectedDept} 
            onChange={(e) => setSelectedDept(e.target.value)}
            className="appearance-none bg-white border border-teal-200 text-teal-800 py-2 pl-10 pr-10 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer font-medium text-sm"
            >
            {departments.map(dept => (
                <option key={dept} value={dept}>{dept}</option>
            ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center px-3 text-teal-500">
                <Layers size={16} />
            </div>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-teal-500">
                <ChevronDown size={16} />
            </div>
        </div>
      </div>

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
                        <button 
                          key={type} 
                          onClick={() => toggleFilter(type)}
                          className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-teal-50 cursor-pointer transition-colors text-left"
                          type="button"
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
                        </button>
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
                <tr>
                  <th 
                    className="p-3 bg-teal-50 cursor-pointer hover:bg-teal-100/50 transition-colors"
                    onClick={() => handleSort('location')}
                  >
                    <div className="flex items-center gap-2">
                      Location
                      {sortConfig.key === 'location' ? (
                         sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />
                      ) : <ArrowUpDown size={12} className="opacity-30" />}
                    </div>
                  </th>
                  <th 
                    className="p-3 bg-teal-50 cursor-pointer hover:bg-teal-100/50 transition-colors"
                    onClick={() => handleSort('severity')}
                  >
                    <div className="flex items-center gap-2">
                      Issue
                      {sortConfig.key === 'severity' ? (
                         sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />
                      ) : <ArrowUpDown size={12} className="opacity-30" />}
                    </div>
                  </th>
                  <th className="p-3 bg-teal-50">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-teal-50">
                {sortedIssues.length === 0 ? (
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
                  sortedIssues.map((issue, idx) => {
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
             <span className="text-xs text-teal-500">{filteredMapMarkers.length} Points</span>
           </div>
           
           <div className="flex-1 relative z-0">
             {filteredMapMarkers.length > 0 ? (
               <MapContainer 
                 center={[filteredMapMarkers[0].lat, filteredMapMarkers[0].lng]} 
                 zoom={15} 
                 style={{ height: "100%", width: "100%" }}
                 scrollWheelZoom={false}
               >
                 <TileLayer
                   attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                   url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                 />
                 {filteredMapMarkers.map((marker, idx) => (
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

        {/* LATENCY HEATMAP PANEL */}
        <div className="col-span-12 bg-white p-6 rounded-xl shadow-sm border border-teal-100">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-teal-800 flex items-center gap-2">
                <Cpu size={20} /> Current Latency Heatmap
            </h3>
            
            {/* Heatmap Toggles */}
            <div className="flex bg-teal-50 p-1 rounded-lg">
                <button
                  onClick={() => setHeatmapMode('LAN')}
                  className={`flex items-center gap-2 px-3 py-1 text-xs font-bold rounded-md transition-all ${
                    heatmapMode === 'LAN' ? 'bg-white text-teal-700 shadow-sm' : 'text-teal-400 hover:text-teal-600'
                  }`}
                >
                  <Network size={14} /> Wired (LAN)
                </button>
                <button
                  onClick={() => setHeatmapMode('WLAN')}
                  className={`flex items-center gap-2 px-3 py-1 text-xs font-bold rounded-md transition-all ${
                    heatmapMode === 'WLAN' ? 'bg-white text-purple-700 shadow-sm' : 'text-teal-400 hover:text-purple-600'
                  }`}
                >
                  <Wifi size={14} /> Wi-Fi (WLAN)
                </button>
            </div>
          </div>
          
          <p className="text-sm text-slate-500 mb-4">Real-time snapshot of {heatmapMode === 'LAN' ? 'wired' : 'wireless'} latency across selected departments.</p>
          
          {filteredHeatmap.length > 0 ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
              {filteredHeatmap.map((probe, i) => (
                <div 
                  key={i} 
                  onClick={() => navigate(`/inspector/${probe.name}`)}
                  className={`p-4 rounded-lg text-center border transition hover:shadow-md cursor-pointer ${
                  probe.color === 'red' ? 'bg-red-50 border-red-200' : 
                  probe.color === 'orange' ? 'bg-orange-50 border-orange-200' :
                  'bg-teal-50 border-teal-200'
                }`}>
                  <div className={`text-xl font-bold ${
                      probe.color === 'red' ? 'text-red-600' : 
                      probe.color === 'orange' ? 'text-orange-600' : 
                      'text-teal-700'
                  }`}>
                    {probe.latency}ms
                  </div>
                  <div className="text-xs text-slate-500 font-medium mt-1 truncate px-1" title={probe.name}>{probe.name}</div>
                </div>
              ))}
            </div>
          ) : (
             <div className="text-center py-8 text-teal-400 italic">No heatmap data available for this filter.</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CommandCenter;