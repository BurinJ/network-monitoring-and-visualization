import React, { useEffect, useState } from 'react';
import { AlertCircle, AlertTriangle, Clock, Server, Info, Wifi, Network } from 'lucide-react';

const AlertHistory = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('All');
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch('http://localhost:5000/api/alerts/history?limit=50')
      .then(res => res.json())
      .then(data => {
        setAlerts(data);
        setLoading(false);
      })
      .catch(err => setLoading(false));
  }, []);

  const formatDate = (isoStr) => {
    if (!isoStr) return '-';
    const date = new Date(isoStr + 'Z'); // Treat stored time as UTC
    return date.toLocaleString();
  };

  // Helper to safely extract interface from details JSON string
  const getInterface = (detailsStr) => {
    try {
      const obj = JSON.parse(detailsStr);
      return obj.interface || '-';
    } catch (e) {
      return '-';
    }
  };

  // FIX: Updated filter logic to match SQLite keys (level, probe_id, message)
  const filteredAlerts = alerts.filter(alert => {
    const matchesFilter = filter === 'All' || alert.level === filter;
    const matchesSearch = 
        (alert.probe_id || '').toLowerCase().includes(search.toLowerCase()) || 
        (alert.message || '').toLowerCase().includes(search.toLowerCase()) ||
        (alert.category || '').toLowerCase().includes(search.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  if (loading) return <div className="p-10 text-teal-600">Loading History...</div>;

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-teal-900 mb-1">Alert Log History</h1>
          <span className="text-sm text-slate-500">Last 50 Events</span>
        </div>
        
        {/* Search & Filter UI */}
        <div className="flex gap-3">
            <input 
              type="text" 
              placeholder="Search logs..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-4 pr-4 py-2 rounded-lg border border-teal-200 focus:outline-none focus:ring-2 focus:ring-teal-500 text-sm w-64"
            />
            <div className="bg-white border border-teal-200 rounded-lg p-1 flex">
             {['All', 'Critical', 'Warning'].map(type => (
               <button
                 key={type}
                 onClick={() => setFilter(type)}
                 className={`px-3 py-1 text-xs font-bold rounded transition-colors ${
                   filter === type 
                     ? 'bg-teal-600 text-white' 
                     : 'text-teal-600 hover:bg-teal-50'
                 }`}
               >
                 {type}
               </button>
             ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-teal-100 overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead className="bg-teal-50 text-xs uppercase text-teal-700">
            <tr>
              <th className="p-4 border-b border-teal-100">Time</th>
              <th className="p-4 border-b border-teal-100">Severity</th>
              <th className="p-4 border-b border-teal-100">Probe</th>
              <th className="p-4 border-b border-teal-100">Interface</th>
              <th className="p-4 border-b border-teal-100">Category</th>
              <th className="p-4 border-b border-teal-100">Message</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-teal-50">
            {filteredAlerts.length === 0 ? (
              <tr><td colSpan="6" className="p-8 text-center text-gray-400">No matching alerts found.</td></tr>
            ) : (
              filteredAlerts.map((alert) => {
                const iface = getInterface(alert.details);
                return (
                  <tr key={alert.id} className="hover:bg-teal-50/50 transition-colors">
                    <td className="p-4 text-xs font-mono text-slate-500 whitespace-nowrap flex items-center gap-2">
                      <Clock size={14} className="text-teal-300" />
                      {formatDate(alert.timestamp)}
                    </td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded text-xs font-bold flex items-center gap-1 w-fit ${
                        alert.level === 'Critical' ? 'bg-red-100 text-red-700' : 
                        alert.level === 'Warning' ? 'bg-orange-100 text-orange-700' : 'bg-blue-100 text-blue-700'
                      }`}>
                        {alert.level === 'Critical' ? <AlertCircle size={12} /> : <AlertTriangle size={12} />}
                        {alert.level}
                      </span>
                    </td>
                    <td className="p-4 text-sm font-bold text-teal-800">
                      {alert.probe_id}
                    </td>
                    <td className="p-4 text-sm text-slate-600">
                       <div className="flex items-center gap-2">
                        {iface === 'WLAN' ? <Wifi size={14} className="text-purple-500"/> : 
                         iface === 'LAN' ? <Network size={14} className="text-blue-500"/> : null}
                        {iface}
                       </div>
                    </td>
                    <td className="p-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      {alert.category}
                    </td>
                    <td className="p-4 text-sm text-slate-700">
                      <div className="font-medium">{alert.message}</div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AlertHistory;