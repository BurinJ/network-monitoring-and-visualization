import React, { useEffect, useState } from 'react';

const NetworkStatus = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('http://localhost:5000/api/network-status')
      .then(res => res.json())
      .then(res => { setData(res); setLoading(false); })
      .catch(err => setLoading(false));
  }, []);

  if (loading) return <div className="p-10 text-center text-teal-600 font-medium">Scanning Network...</div>;

  const buildings = data?.buildings || { lan: [], wlan: [] };

  const StatusCard = ({ item }) => (
    <div className="p-4 border border-teal-100 rounded-xl flex justify-between items-center hover:shadow-lg transition bg-white group">
      <div>
        <span className="font-bold text-slate-700 block group-hover:text-teal-700 transition">{item.name}</span>
        <span className="text-xs text-gray-400">DNS: {item.dns}ms</span>
      </div>
      <div className="text-right">
        <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider
          ${item.color === 'green' ? 'bg-emerald-100 text-emerald-700' : 
            item.color === 'orange' ? 'bg-orange-100 text-orange-700' : 'bg-red-100 text-red-700'}`}>
          <span className={`w-2 h-2 rounded-full ${item.color === 'green' ? 'bg-emerald-500' : item.color === 'orange' ? 'bg-orange-500' : 'bg-red-500'}`}></span>
          {item.status}
        </span>
        <div className="text-xs text-gray-500 mt-1 font-mono">{item.latency}ms</div>
      </div>
    </div>
  );

  return (
    <div className="p-8 bg-teal-50 min-h-screen">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-teal-900 mb-2">Connectivity Status</h1>
        <p className="text-teal-600">Real-time connectivity monitoring</p>
      </div>

      {/* Main Status Banner */}
      <div className={`max-w-6xl mx-auto border-2 rounded-2xl p-8 flex items-center justify-between mb-12 shadow-sm bg-white
        ${data?.overall === 'Operational' ? 'border-emerald-400' : 'border-red-400'}`}>
        <div>
          <h2 className={`text-3xl font-bold ${data?.overall === 'Operational' ? 'text-emerald-700' : 'text-red-700'}`}>
             {data?.overall === 'Operational' ? '✅ System Healthy' : '⚠️ Service Disruption'}
          </h2>
          <p className="mt-2 text-slate-600">
            {data?.overall === 'Operational' 
              ? 'All LAN and WLAN interfaces are performing within optimal parameters.' 
              : 'Attention required: Issues detected in one or more network segments.'}
          </p>
        </div>
      </div>

      <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* LAN SECTION */}
        <div className="bg-white/50 p-6 rounded-2xl border border-teal-100">
          <h3 className="text-xl font-bold text-teal-800 mb-6 flex items-center gap-3 border-b border-teal-200 pb-4">
            <span className="p-2 bg-teal-100 rounded-lg text-teal-600">🌐</span> Ethernet (LAN)
          </h3>
          <div className="flex flex-col gap-3">
            {buildings.lan.length === 0 && <div className="text-gray-400 italic text-center py-4">No active LAN monitors found.</div>}
            {buildings.lan.map((place) => (
              <StatusCard key={place.id + 'lan'} item={place} />
            ))}
          </div>
        </div>

        {/* WLAN SECTION */}
        <div className="bg-white/50 p-6 rounded-2xl border border-teal-100">
          <h3 className="text-xl font-bold text-teal-800 mb-6 flex items-center gap-3 border-b border-teal-200 pb-4">
            <span className="p-2 bg-teal-100 rounded-lg text-teal-600">📶</span> Wi-Fi (WLAN)
          </h3>
          <div className="flex flex-col gap-3">
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