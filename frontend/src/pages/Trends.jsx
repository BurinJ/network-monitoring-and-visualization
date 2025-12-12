import React, { useEffect, useState } from 'react';

const Trends = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5000/api/trends')
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);

  if (!data) return <div className="p-10">Generating Forecasts...</div>;

  return (
    <div className="p-6 bg-gray-50 min-h-screen">
      <h1 className="text-2xl font-bold text-slate-800 mb-6">Trends & Forecasts</h1>
      
      {/* Forecast Chart */}
      <div className="bg-white p-6 rounded-xl shadow-sm border mb-6">
        <div className="flex justify-between mb-4">
          <h3 className="font-bold text-gray-700">Bandwidth Load Prediction (Next 10 Days)</h3>
          <span className="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">AI Model: LSTM</span>
        </div>
        
        {/* Simple visual bar chart for forecast */}
        <div className="flex items-end space-x-2 h-64 border-b border-l p-4">
          {data.forecast.map((day, i) => (
            <div key={i} className="flex-1 flex flex-col items-center group">
               <div 
                 className="w-full bg-blue-500 opacity-80 rounded-t hover:opacity-100 transition-all" 
                 style={{ height: `${day.predicted_load}%` }}
               ></div>
               <span className="text-xs text-gray-500 mt-2 rotate-45 origin-left">{day.time}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Heatmap List */}
      <div className="bg-white p-6 rounded-xl shadow-sm border">
        <h3 className="font-bold text-gray-700 mb-4">Current Problem Hotspots</h3>
        <div className="grid grid-cols-4 gap-4">
           {data.heatmap.map((probe, i) => (
             <div key={i} className={`p-4 rounded text-center border ${
               probe.color === 'red' ? 'bg-red-50 border-red-200' : 'bg-gray-50 border-gray-200'
             }`}>
               <div className={`text-xl font-bold ${probe.color === 'red' ? 'text-red-600' : 'text-gray-600'}`}>
                 {probe.latency}ms
               </div>
               <div className="text-sm text-gray-500">{probe.name}</div>
             </div>
           ))}
        </div>
      </div>
    </div>
  );
};

export default Trends;