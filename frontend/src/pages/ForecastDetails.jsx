import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { TrendingUp, ArrowLeft, Eye, EyeOff } from 'lucide-react';
import { API_BASE_URL } from '../config';

const ForecastDetails = () => {
  const { probeId } = useParams();
  const [data, setData] = useState(null);
  const [showRF, setShowRF] = useState(true);
  const [showLSTM, setShowLSTM] = useState(true);

  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/forecast/${probeId}`)
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, [probeId]);

  if (!data) return <div className="p-10 text-teal-600 animate-pulse">Calculating Forecast...</div>;

  // Helper for the combined graph
  const ForecastGraph = ({ history, forecast }) => {
    const height = 300;
    const width = 100; // percent
    const graphWidth = 800; // internal SVG units
    
    // Combine data for scaling: [timestamp, value]
    const histVals = history.map(d => d[1]);
    const foreVals = forecast.map(d => Math.max(d.rf_value, d.lstm_value));
    const maxVal = Math.max(...histVals, ...foreVals, 10);
    
    const totalPoints = history.length + forecast.length;
    
    // X Coordinate Helper
    const getX = (i) => (i / (totalPoints - 1)) * graphWidth;
    // Y Coordinate Helper
    const getY = (val) => height - (val / maxVal) * height;

    const histPoints = history.map((d, i) => `${getX(i)},${getY(d[1])}`).join(' ');

    // Last history point index
    const lastHistIdx = history.length - 1;
    const lastHistX = getX(lastHistIdx);
    const lastHistY = getY(history[lastHistIdx][1]);

    const rfPoints = forecast.map((d, i) => `${getX(lastHistIdx + 1 + i)},${getY(d.rf_value)}`).join(' ');
    const lstmPoints = forecast.map((d, i) => `${getX(lastHistIdx + 1 + i)},${getY(d.lstm_value)}`).join(' ');
    
    // Connect forecast to history
    const rfPath = `${lastHistX},${lastHistY} ${rfPoints}`;
    const lstmPath = `${lastHistX},${lastHistY} ${lstmPoints}`;

    // Combine labels for X-axis
    // Ensure labels are distinct and handle time rollover correctly
    const allLabels = [
        ...history.map(d => new Date(d[0] * 1000).toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit', hour12: false})),
        ...forecast.map(d => d.time)
    ];

    return (
      <div className="relative border-b border-l border-slate-200 bg-white rounded-lg p-4 shadow-sm overflow-hidden h-[340px] pl-12">
         {/* Y-Axis Labels */}
         <div className="absolute left-0 top-4 bottom-10 w-10 flex flex-col justify-between text-[10px] text-gray-400 font-mono text-right pr-2">
            <span>{Math.round(maxVal)}</span>
            <span>{Math.round(maxVal * 0.75)}</span>
            <span>{Math.round(maxVal * 0.5)}</span>
            <span>{Math.round(maxVal * 0.25)}</span>
            <span>0</span>
        </div>

        <div className="relative h-[300px] w-full">
            <svg viewBox={`0 0 ${graphWidth} ${height}`} className="w-full h-full overflow-visible" preserveAspectRatio="none">
            {/* Grid Lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => (
                <line key={i} x1="0" y1={height * pct} x2={graphWidth} y2={height * pct} stroke="#eee" strokeDasharray="4" vectorEffect="non-scaling-stroke" />
            ))}
            
            {/* Divider Line (Now vs Future) */}
            <line x1={lastHistX} y1="0" x2={lastHistX} y2={height} stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4" vectorEffect="non-scaling-stroke" />
            <text x={lastHistX + 5} y="15" className="text-xs fill-slate-400 font-bold">NOW</text>

            {/* History Line (Solid Slate) */}
            <polyline fill="none" stroke="#64748b" strokeWidth="2" points={histPoints} vectorEffect="non-scaling-stroke" />
            
            {/* RF Forecast (Teal) */}
            {showRF && <polyline fill="none" stroke="#0d9488" strokeWidth="2" points={rfPath} vectorEffect="non-scaling-stroke" />}
            
            {/* LSTM Forecast (Purple) */}
            {showLSTM && <polyline fill="none" stroke="#9333ea" strokeWidth="2" strokeDasharray="5,5" points={lstmPath} vectorEffect="non-scaling-stroke" />}

            {/* Interactive History Points */}
            {history.map((d, i) => (
                <g key={`h-${i}`} className="group">
                    {/* Invisible Hit Area for better hover target */}
                    <rect 
                        x={getX(i) - 5} y="0" width="10" height={height} 
                        fill="transparent" 
                        className="cursor-pointer" 
                    >
                         <title>{`${new Date(d[0] * 1000).toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit', hour12: false})}\nHistory: ${Math.round(d[1])} Mbps`}</title>
                    </rect>
                    {/* Visible Dot */}
                    <circle 
                        cx={getX(i)} cy={getY(d[1])} r="3" 
                        fill="#64748b" 
                        className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"
                    />
                </g>
            ))}

            {/* Interactive Forecast Points */}
            {forecast.map((d, i) => {
                const x = getX(lastHistIdx + 1 + i);
                const yRF = getY(d.rf_value);
                const yLSTM = getY(d.lstm_value);
                
                return (
                <g key={`f-${i}`} className="group">
                    {/* Invisible Hit Area for better hover target */}
                    <rect 
                        x={x - 5} y="0" width="10" height={height} 
                        fill="transparent" 
                        className="cursor-pointer" 
                    >
                        <title>{`${d.time}\nRF: ${d.rf_value} Mbps\nLSTM: ${d.lstm_value} Mbps`}</title>
                    </rect>
                    {/* RF Dot */}
                    {showRF && (
                        <circle cx={x} cy={yRF} r="4" fill="#0d9488" className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                    )}
                    {/* LSTM Dot */}
                    {showLSTM && (
                        <circle cx={x} cy={yLSTM} r="4" fill="#9333ea" className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                    )}
                </g>
                )
            })}
            </svg>

            {/* X Axis Labels (HTML) */}
            <div className="absolute left-0 right-0 top-[310px] h-6 pointer-events-none">
                {allLabels.map((label, i) => {
                    // Show label every ~6 points to prevent overcrowding
                    if (i % 6 !== 0) return null;
                    const leftPct = (i / (totalPoints - 1)) * 100;
                    return (
                        <div 
                            key={i} 
                            className="absolute text-[10px] text-gray-400 transform -translate-x-1/2 -translate-y-full"
                            style={{ left: `${leftPct}%` }}
                        >
                            {label}
                        </div>
                    );
                })}
            </div>
        </div>
        
        {/* Legend */}
        <div className="flex justify-between mt-8 text-sm text-slate-500 font-mono pl-2">
            <span>-24 Hours</span>
            <span>Now</span>
            <span>+24 Hours</span>
        </div>
      </div>
    );
  };

  return (
    <div className="p-8 bg-teal-50 min-h-screen">
      <div className="flex items-center gap-4 mb-8">
        <Link to={`/inspector/${probeId}`} className="p-2 bg-white rounded-full shadow-sm text-teal-600 hover:text-teal-800 transition">
            <ArrowLeft size={24} />
        </Link>
        <div>
            <h1 className="text-2xl font-bold text-teal-900">Traffic Forecast: {data.probe_name}</h1>
            <p className="text-teal-600 text-sm">AI Prediction based on past 24h behavior</p>
        </div>
      </div>

      <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100 mb-8">
        <div className="flex flex-wrap justify-between items-center mb-6 gap-4">
            <h3 className="font-bold text-teal-800">Traffic Load Analysis</h3>
            
            <div className="flex gap-4 text-sm">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-slate-500 rounded-full"></div> 
                    <span className="text-slate-600">History</span>
                </div>
                
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
        
        <ForecastGraph history={data.history} forecast={data.forecast} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100">
            <h3 className="font-bold text-teal-800 mb-4">Predicted Peak Load</h3>
            {(() => {
                const maxLoad = Math.max(...data.forecast.map(d => d.rf_value));
                const maxTime = data.forecast.find(d => d.rf_value === maxLoad)?.time;
                return (
                    <div>
                        <div className="text-4xl font-bold text-teal-600">{maxLoad} Mbps</div>
                        <div className="text-sm text-slate-500">Expected at {maxTime}</div>
                    </div>
                );
            })()}
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100">
            <h3 className="font-bold text-teal-800 mb-4">Anomaly Probability</h3>
            <div className="flex items-center gap-4">
                <div className={`p-3 rounded-full ${data.forecast.some(d => Math.abs(d.rf_value - d.lstm_value) > 200) ? 'bg-orange-100 text-orange-600' : 'bg-green-100 text-green-600'}`}>
                    <TrendingUp size={24} />
                </div>
                <div>
                    <div className="font-bold text-slate-700">
                        {data.forecast.some(d => Math.abs(d.rf_value - d.lstm_value) > 200) ? 'High Divergence' : 'Consistent'}
                    </div>
                    <div className="text-xs text-slate-400">
                        Model agreement indicates confidence
                    </div>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastDetails;