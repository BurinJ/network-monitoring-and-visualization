import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
// Import Leaflet components (Error handling if not installed)
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';
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
  const radius = 50; // Increased radius for bigger donuts
  const circumference = 2 * Math.PI * radius;
  const activeOffset = circumference - (percentage / 100) * circumference;
  
  return (
    <div className="flex flex-col items-center group relative">
      <div className="relative w-32 h-32"> {/* Increased container size */}
        <svg className="w-full h-full transform -rotate-90">
          {/* Background Circle (Represents the total or 'bad' part if 'good' is on top) */}
          <circle
            cx="64" cy="64" r={radius} // Adjusted center
            stroke="currentColor" strokeWidth="20" fill="transparent" 
            className="text-red-100 hover:opacity-60 transition-opacity duration-300 ease-in-out cursor-pointer"
          >
            <title>{`Issues: ${issues}`}</title>
          </circle>
          
          {/* Foreground Circle (Active/Good Value) */}
          <circle
            cx="64" cy="64" r={radius} // Adjusted center
            stroke="currentColor" strokeWidth="20" fill="transparent" 
            strokeDasharray={circumference}
            strokeDashoffset={activeOffset}
            strokeLinecap="butt" // Changed from 'round' to 'butt' for flat ends
            className={`${colorClass} hover:opacity-60 transition-opacity duration-300 ease-in-out cursor-pointer`}
          >
             <title>{`Active: ${active}`}</title>
          </circle>
        </svg>
        
        {/* Center Text */}
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

  useEffect(() => {
    // Fetch aggregated stats
    fetch('http://localhost:5000/api/command-center')
      .then(res => res.json())
      .then(setStats)
      .catch(console.error);

    // Fetch detailed status for charts
    fetch('http://localhost:5000/api/network-status')
      .then(res => res.json())
      .then(setNetworkData)
      .catch(console.error);
  }, []);

  if (!stats) return <div className="p-10 text-teal-600">Loading Dashboard...</div>;

  // Calculate Chart Data
  const lanProbes = networkData?.buildings?.lan || [];
  const wlanProbes = networkData?.buildings?.wlan || [];
  
  const lanActive = lanProbes.filter(p => p.color !== 'red').length;
  const wlanActive = wlanProbes.filter(p => p.color !== 'red').length;

  // Helper to determine circle style based on status color
  const getCircleOptions = (color) => {
    const colorMap = {
      green: '#10b981', // emerald-500
      orange: '#f97316', // orange-500
      red: '#ef4444',    // red-500
    };
    const hex = colorMap[color] || colorMap.green;
    
    return {
      color: hex,       // Stroke color
      fillColor: hex,   // Fill color
      fillOpacity: 0.6,
      weight: 2,        // Stroke width
      radius: 4         // Radius in pixels
    };
  };

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <h1 className="text-2xl font-bold text-teal-900 mb-6">Monitoring Dashboard</h1>

      <div className="grid grid-cols-12 gap-6">
        {/* Top Cards Row */}
        
        {/* Active Alerts (Smaller, 4 columns) */}
        <div className="col-span-12 md:col-span-4 mb-2">
           <div className="bg-white rounded-xl border border-teal-100 p-4 shadow-sm flex flex-col justify-center h-full">
             <span className="text-teal-500 text-sm font-semibold uppercase">Active Alerts</span>
             <div className="text-5xl font-bold text-red-500 mt-2">{stats.alerts}</div>
             <div className="text-xs text-red-400 mt-1">Requires immediate attention</div>
           </div>
        </div>
           
        {/* Services Available Charts (Larger, 8 columns) */}
        <div className="col-span-12 md:col-span-8 mb-2">
           <div className="bg-white rounded-xl border border-teal-100 p-4 shadow-sm flex flex-col justify-between h-full">
             <span className="text-teal-500 text-sm font-semibold uppercase mb-2">Services Available</span>
             <div className="flex items-center justify-around h-full">
                <MiniDonut 
                  label="LAN" 
                  active={lanActive} 
                  total={lanProbes.length} 
                  colorClass="text-blue-500" 
                />
                <div className="w-px h-24 bg-teal-50 mx-4"></div>
                <MiniDonut 
                  label="WLAN" 
                  active={wlanActive} 
                  total={wlanProbes.length} 
                  colorClass="text-purple-500" 
                />
             </div>
           </div>
        </div>

        {/* Priority Issues */}
        <div className="col-span-8 bg-white rounded-xl shadow-sm border border-teal-100 p-6 flex flex-col">
          <h3 className="font-bold mb-4 flex items-center gap-2 text-teal-800">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span> Priority Issues
          </h3>
          <div className="flex-1 overflow-auto">
            <table className="w-full text-left">
              <thead className="bg-teal-50 text-xs uppercase text-teal-600">
                <tr><th className="p-3">Location</th><th className="p-3">Issue</th><th className="p-3">Action</th></tr>
              </thead>
              <tbody className="divide-y divide-teal-50">
                {stats.priority_issues.length === 0 ? (
                   <tr><td colSpan="3" className="p-4 text-center text-teal-400 italic">No active issues. System Healthy.</td></tr>
                ) : (
                  stats.priority_issues.map((issue, idx) => {
                    const cleanLocation = issue.location.replace(/ \((LAN|WLAN)\)$/, '');
                    return (
                      <tr key={idx} className="hover:bg-teal-50/50">
                        <td className="p-3 font-medium text-teal-900">{issue.location}</td>
                        <td className="p-3">
                          <span className={`px-2 py-1 rounded text-xs font-bold ${issue.severity === 'High' ? 'bg-red-100 text-red-700' : 'bg-orange-100 text-orange-700'}`}>
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
        <div className="col-span-4 bg-white rounded-xl border border-teal-100 overflow-hidden shadow-sm flex flex-col">
           <div className="p-4 border-b border-teal-50 flex justify-between items-center">
             <h3 className="font-bold text-teal-800">Probe Locations</h3>
             <span className="text-xs text-teal-500">{stats.map_markers?.length || 0} Points</span>
           </div>
           
           <div className="flex-1 min-h-[300px] relative z-0">
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
                     pathOptions={{
                        ...getCircleOptions(marker.color),
                        className: 'hover:opacity-60 transition-opacity duration-300 ease-in-out cursor-pointer' // Add hover effect class here
                     }}
                     eventHandlers={{
                        click: () => navigate(`/inspector/${marker.name}`)
                     }}
                   >
                     <Tooltip direction="top" offset={[0, -10]} opacity={1}>
                       <div className="text-center">
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