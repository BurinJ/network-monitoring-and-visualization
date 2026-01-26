import React, { useEffect, useState } from 'react';
import { TrendingUp, Cpu, Zap, Eye, EyeOff, HelpCircle } from 'lucide-react';

const Trends = () => {
  const [data, setData] = useState(null);
  const [showRF, setShowRF] = useState(true);
  const [showLSTM, setShowLSTM] = useState(true);
  const [showInfo, setShowInfo] = useState(false);

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
    
    const height = 250; 
    const graphWidth = 600; // Internal SVG coord width
    
    // Y Scale: 0 to 100%
    const getY = (val) => height - (val / 100) * height;
    
    const rfPoints = data.map((d, i) => {
      const x = (i / (data.length - 1)) * graphWidth;
      return `${x},${getY(d.rf_load)}`;
    }).join(' ');

    const lstmPoints = data.map((d, i) => {
      const x = (i / (data.length - 1)) * graphWidth;
      return `${x},${getY(d.lstm_load)}`;
    }).join(' ');

    return (
      <div className="relative h-80 w-full pl-10">
        {/* Y-Axis Labels */}
        <div className="absolute left-0 top-0 h-[250px] w-8 flex flex-col justify-between text-[10px] text-gray-400 font-mono text-right pr-2">
            <span>100%</span>
            <span>75%</span>
            <span>50%</span>
            <span>25%</span>
            <span>0%</span>
        </div>

        <svg viewBox={`0 0 ${graphWidth} ${height}`} className="w-full h-[250px] overflow-visible" preserveAspectRatio="none">
          {/* Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => (
             <line key={i} x1="0" y1={height * pct} x2={graphWidth} y2={height * pct} stroke="#e2e8f0" strokeDasharray="4" vectorEffect="non-scaling-stroke" />
          ))}

          {/* Random Forest Line (Teal) */}
          {showRF && <polyline fill="none" stroke="#0d9488" strokeWidth="2" points={rfPoints} vectorEffect="non-scaling-stroke" />}
          
          {/* LSTM Line (Purple) */}
          {showLSTM && <polyline fill="none" stroke="#9333ea" strokeWidth="2" strokeDasharray="5,5" points={lstmPoints} vectorEffect="non-scaling-stroke" />}
          
          {/* Interactive Points Group */}
          {data.map((d, i) => {
            const x = (i / (data.length - 1)) * graphWidth;
            const yRF = getY(d.rf_load);
            const yLSTM = getY(d.lstm_load);
            
            return (
              <g key={i}>
                {/* Invisible hover bar for easier interaction 
                <rect 
                    x={x - 10} y="0" width="20" height={height} 
                    fill="transparent" 
                    className="group hover:fill-teal-50/30 cursor-pointer transition-colors" 
                />
                */}
                
                {/* Random Forest Point */}
                {showRF && (
                  <circle 
                    cx={x} cy={yRF} r="4" 
                    fill="#0d9488" 
                    className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
                  >
                    <title>{`${d.time}\nRandom Forest: ${d.rf_load}% Load\n(Based on historic patterns)`}</title>
                  </circle>
                )}
                
                {/* LSTM Point */}
                {showLSTM && (
                  <circle 
                    cx={x} cy={yLSTM} r="4" 
                    fill="#9333ea" 
                    className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
                  >
                    <title>{`${d.time}\nLSTM: ${d.lstm_load}% Load\n(Based on recent trends)`}</title>
                  </circle>
                )}
              </g>
            );
          })}
        </svg>

        {/* X Axis Labels (HTML for non-stretched text) */}
        <div className="absolute left-10 right-0 top-[260px] h-6">
           {data.map((d, i) => {
              // Show every 4th label to prevent overcrowding
              if (i % 4 !== 0) return null;
              const leftPct = (i / (data.length - 1)) * 100;
              return (
                  <div 
                      key={i} 
                      className="absolute text-[10px] text-gray-400 transform -translate-x-1/2"
                      style={{ left: `${leftPct}%` }}
                  >
                      {d.time}
                  </div>
              );
           })}
        </div>
      </div>
    );
  };

  return (
    <div className="p-8 bg-teal-50 min-h-screen">
      <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-teal-900">Global Network Trends</h1>
            <p className="text-teal-600">AI-driven capacity planning and load forecasting</p>
          </div>
          <button 
            onClick={() => setShowInfo(!showInfo)}
            className="flex items-center gap-2 text-sm font-medium text-teal-700 bg-white px-4 py-2 rounded-lg border border-teal-200 hover:bg-teal-50 transition-colors shadow-sm"
          >
            <HelpCircle size={18} />
            How this works
          </button>
      </div>
      
      {/* Help Info Box */}
      {showInfo && (
        <div className="bg-white p-6 rounded-xl border-l-4 border-purple-500 shadow-sm mb-8 animate-in slide-in-from-top-2">
            <div className="flex justify-between items-start">
                <h3 className="font-bold text-lg text-slate-800 mb-4">Understanding Forecast Models</h3>
                <button onClick={() => setShowInfo(false)} className="text-slate-400 hover:text-slate-600">Close</button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-600">
                <div className="bg-teal-50 p-4 rounded-lg border border-teal-100">
                    <strong className="text-teal-800 flex items-center gap-2 mb-2">
                        <div className="w-3 h-3 bg-teal-600 rounded-full"></div> 
                        Random Forest (Long-Term Patterns)
                    </strong>
                    <p>This model learns routine schedules. For example, it knows that "Monday mornings are usually busy" and "Weekends are quiet." It is less sensitive to sudden, temporary spikes.</p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg border border-purple-100">
                    <strong className="text-purple-800 flex items-center gap-2 mb-2">
                        <div className="w-3 h-3 border-2 border-purple-600 border-dashed rounded-full"></div> 
                        LSTM (Short-Term Sequences)
                    </strong>
                    <p>This model looks at the last 24 hours to predict the next few hours. If traffic has been unexpectedly rising all morning, LSTM will predict it continues to rise, even if that's unusual for a Tuesday.</p>
                </div>
            </div>
        </div>
      )}
      
      {/* Forecast Chart */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100 mb-8">
        <div className="flex flex-col md:flex-row justify-between md:items-center mb-6 gap-4">
          <div>
            <h3 className="font-bold text-teal-800 text-lg flex items-center gap-2">
                <Zap size={20} className="text-yellow-500" /> 
                Predicted Network Load (Next 24 Hours)
            </h3>
            <p className="text-xs text-slate-400 mt-1">Y-Axis represents % of total estimated network capacity.</p>
          </div>
          
          <div className="flex gap-3 text-sm">
            <button 
              onClick={() => setShowRF(!showRF)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all ${
                  showRF ? 'bg-teal-50 text-teal-800 border-teal-200' : 'bg-white text-slate-400 border-slate-100'
              }`}
            >
              <div className={`w-3 h-3 rounded-full ${showRF ? 'bg-teal-600' : 'bg-slate-300'}`}></div>
              Random Forest
              {showRF ? <Eye size={14} /> : <EyeOff size={14} />}
            </button>
            
            <button 
              onClick={() => setShowLSTM(!showLSTM)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all ${
                  showLSTM ? 'bg-purple-50 text-purple-800 border-purple-200' : 'bg-white text-slate-400 border-slate-100'
              }`}
            >
              <div className={`w-3 h-3 rounded-full border-2 ${showLSTM ? 'border-purple-600' : 'border-slate-300'} border-dashed`}></div>
              LSTM
              {showLSTM ? <Eye size={14} /> : <EyeOff size={14} />}
            </button>
          </div>
        </div>
        
        <LineGraph data={data.forecast} />
        
        <div className="mt-2 pt-4 border-t border-slate-100 text-center flex justify-center gap-8 text-xs text-slate-400">
             <span>Prediction generated at: <span className="font-mono text-slate-600">{new Date().toLocaleTimeString()}</span></span>
             <span>Input History: <span className="font-mono text-slate-600">Last 24 Hours</span></span>
        </div>
      </div>
    </div>
  );
};

export default Trends;