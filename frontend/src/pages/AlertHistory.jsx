import React, { useEffect, useState } from 'react';
import { AlertCircle, AlertTriangle, Clock, Server, Info } from 'lucide-react';

const AlertHistory = () => {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

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
    const date = new Date(isoStr + 'Z'); // Ensure UTC
    return date.toLocaleString();
  };

  if (loading) return <div className="p-10 text-teal-600">Loading History...</div>;

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-teal-900">Alert Log History</h1>
        <span className="text-sm text-slate-500">Last 50 Events</span>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-teal-100 overflow-hidden">
        <table className="w-full text-left border-collapse">
          <thead className="bg-teal-50 text-xs uppercase text-teal-700">
            <tr>
              <th className="p-4 border-b border-teal-100">Time</th>
              <th className="p-4 border-b border-teal-100">Severity</th>
              <th className="p-4 border-b border-teal-100">Probe</th>
              <th className="p-4 border-b border-teal-100">Category</th>
              <th className="p-4 border-b border-teal-100">Message</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-teal-50">
            {alerts.length === 0 ? (
              <tr><td colSpan="5" className="p-8 text-center text-gray-400">No alerts recorded yet.</td></tr>
            ) : (
              alerts.map((alert) => (
                <tr key={alert.id} className="hover:bg-teal-50/50 transition-colors">
                  <td className="p-4 text-sm text-slate-600 whitespace-nowrap flex items-center gap-2">
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
                  <td className="p-4 text-sm font-medium text-teal-900 flex items-center gap-2">
                    <Server size={14} className="text-teal-300" />
                    {alert.probe_id}
                  </td>
                  <td className="p-4 text-sm text-slate-600">{alert.category}</td>
                  <td className="p-4 text-sm text-slate-700">
                    <div className="font-medium">{alert.message}</div>
                    {alert.details && <div className="text-xs text-slate-400 mt-0.5">{alert.details}</div>}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AlertHistory;