import React, { useEffect, useState } from 'react';
import { TrendingUp, Cpu, Zap } from 'lucide-react';

const Trends = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    fetch('http://localhost:5000/api/trends')
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);

  if (!data) return <div className="p-10 text-teal-600 animate-pulse">Generating AI Forecasts...</div>;

  // Helper for multi-line graph
  const LineGraph = ({ data }) => {
    if (!data || data.length === 0) return null;
    
    const height = 200;
    const width = 100; // percent
    
    // Y Scale: 0 to 100%
    const getY = (val) => height - (val / 100) * height;
    
    const rfPoints = data.map((d, i) => {
      const x = (i / (data.length - 1)) * 600; // scaled width
      return `${x},${getY(d.rf_load)}`;
    }).join(' ');

    const lstmPoints = data.map((d, i) => {
      const x = (i / (data.length - 1)) * 600;
      return `${x},${getY(d.lstm_load)}`;
    }).join(' ');

    return (
      <div className="relative h-64 w-full">
        <svg viewBox={`0 0 600 ${height}`} className="w-full h-full overflow-visible" preserveAspectRatio="none">
          {/* Grid Lines */}
          <line x1="0" y1={height*0.25} x2="600" y2={height*0.25} stroke="#e2e8f0" strokeDasharray="4" />
          <line x1="0" y1={height*0.5} x2="600" y2={height*0.5} stroke="#e2e8f0" strokeDasharray="4" />
          <line x1="0" y1={height*0.75} x2="600" y2={height*0.75} stroke="#e2e8f0" strokeDasharray="4" />

          {/* Random Forest Line (Teal) */}
          <polyline fill="none" stroke="#0d9488" strokeWidth="2" points={rfPoints} />
          {/* LSTM Line (Purple) */}
          <polyline fill="none" stroke="#9333ea" strokeWidth="2" strokeDasharray="5,5" points={lstmPoints} />
          
          {/* Interactive Points */}
          {data.map((d, i) => {
            const x = (i / (data.length - 1)) * 600;
            const yRF = getY(d.rf_load);
            const yLSTM = getY(d.lstm_load);
            
            return (
              <g key={i}>
                {/* Random Forest Point */}
                <circle 
                  cx={x} cy={yRF} r="3" 
                  fill="#0d9488" 
                  className="hover:r-5 transition-all cursor-pointer opacity-0 hover:opacity-100"
                >
                  <title>{`${d.time} - RF Prediction: ${d.rf_load}%`}</title>
                </circle>
                
                {/* LSTM Point */}
                <circle 
                  cx={x} cy={yLSTM} r="3" 
                  fill="#9333ea" 
                  className="hover:r-5 transition-all cursor-pointer opacity-0 hover:opacity-100"
                >
                  <title>{`${d.time} - LSTM Prediction: ${d.lstm_load}%`}</title>
                </circle>
              </g>
            );
          })}
          
          {/* X Axis Labels */}
          {data.filter((_, i) => i % 4 === 0).map((d, i) => (
             <text key={i} x={(i*4 / (data.length - 1)) * 600} y={height + 20} className="text-[10px] fill-slate-400" textAnchor="middle">
               {d.time}
             </text>
          ))}
        </svg>
      </div>
    );
  };

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <h1 className="text-2xl font-bold text-teal-900 mb-6">Trends & Forecasts</h1>
      
      {/* Forecast Chart */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100 mb-6">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="font-bold text-teal-800 text-lg">Network Load Prediction (Next 24 Hours)</h3>
            <p className="text-sm text-slate-500">Comparison of Statistical vs. Sequential AI Models</p>
          </div>
          
          <div className="flex gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-teal-600 rounded-full"></div>
              <span className="font-bold text-teal-700">Random Forest</span>
              <span className="text-xs text-slate-400">(Pattern Based)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 border-2 border-purple-600 border-dashed rounded-full"></div>
              <span className="font-bold text-purple-700">LSTM</span>
              <span className="text-xs text-slate-400">(Sequence Based)</span>
            </div>
          </div>
        </div>
        
        <LineGraph data={data.forecast} />
      </div>

      {/* Heatmap List */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100">
        <h3 className="font-bold text-teal-800 mb-4">Current Probe Latency Map</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
           {data.heatmap.map((probe, i) => (
             <div key={i} className={`p-4 rounded-lg text-center border transition hover:shadow-md ${
               probe.color === 'red' ? 'bg-red-50 border-red-200' : 'bg-teal-50 border-teal-200'
             }`}>
               <div className={`text-xl font-bold ${probe.color === 'red' ? 'text-red-600' : 'text-teal-700'}`}>
                 {probe.latency}ms
               </div>
               <div className="text-xs text-teal-500 font-medium mt-1">{probe.name}</div>
             </div>
           ))}
        </div>
      </div>
    </div>
  );
};

export default Trends;