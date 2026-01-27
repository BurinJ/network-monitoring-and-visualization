import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { TrendingUp, ArrowLeft, Eye, EyeOff, Network, Globe, Activity, ArrowDown, ArrowUp as ArrowUpIcon, Layers, ChevronDown } from 'lucide-react';
import { API_BASE_URL } from '../config';

const ForecastDetails = () => {
  const { probeId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [showRF, setShowRF] = useState(true);
  const [showLSTM, setShowLSTM] = useState(true);

  // Selector State
  const [allProbes, setAllProbes] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [selectedDept, setSelectedDept] = useState('All');

  // View State
  const [selectedInterface, setSelectedInterface] = useState('lan'); // 'lan' or 'wlan'
  const [selectedMetric, setSelectedMetric] = useState('ext_down'); // 'ext_down', 'ext_up', 'int_down', 'int_up', 'ping_ext', 'ping_int'

  // 1. Fetch Probe List (Includes department info)
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/settings/probes`)
      .then(res => res.json())
      .then(data => {
        setAllProbes(data);
        
        // Extract unique departments
        const depts = Array.from(new Set(data.map(p => p.department || 'Undefined')));
        setDepartments(['All', ...depts.sort()]);
        
        // Sync selected department with current probeId if needed
        const current = data.find(p => p.name === probeId);
        if (current) {
            setSelectedDept(current.department || 'Undefined');
        }
      })
      .catch(console.error);
  }, []);

  // Update selected department when probeId changes
  useEffect(() => {
    if (allProbes.length > 0) {
        const current = allProbes.find(p => p.name === probeId);
        if (current) {
            setSelectedDept(current.department || 'Undefined');
        }
    }
  }, [probeId, allProbes]);

  // 2. Fetch Forecast Data
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    setData(null);
    fetch(`${baseUrl}/forecast/${probeId}`)
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, [probeId]);

  const handleProbeChange = (e) => {
    navigate(`/forecast/${e.target.value}`);
  };

  const handleDeptChange = (e) => {
    const newDept = e.target.value;
    setSelectedDept(newDept);
    
    // Optional: Switch to the first probe in that department
    const firstProbe = allProbes.find(p => (newDept === 'All' || (p.department || 'Undefined') === newDept));
    if (firstProbe && firstProbe.name !== probeId) {
        navigate(`/forecast/${firstProbe.name}`);
    }
  };

  // Filter probes for dropdown
  const filteredProbes = allProbes.filter(p => selectedDept === 'All' || (p.department || 'Undefined') === selectedDept);

  if (!data) return <div className="p-10 text-teal-600 animate-pulse">Calculating Multi-Metric Forecast...</div>;

  // Construct key to access data: e.g. "lan_ext_down"
  const activeKey = `${selectedInterface}_${selectedMetric}`;
  const activeData = data.data[activeKey] || { history: [], forecast: [], max: 100 };

  // Helper for the combined graph
  const ForecastGraph = ({ history, forecast, max }) => {
    const height = 300;
    const width = 100; 
    const graphWidth = 800; 
    
    // Scale Y based on max value of this specific series
    // Ensure we don't divide by zero
    const maxVal = Math.max(max, 10);
    
    const totalPoints = history.length + forecast.length;
    
    const getX = (i) => (i / (totalPoints - 1)) * graphWidth;
    const getY = (val) => height - (val / maxVal) * height;

    const histPoints = history.map((d, i) => `${getX(i)},${getY(d[1])}`).join(' ');

    const lastHistIdx = history.length - 1;
    const lastHistX = getX(lastHistIdx);
    const lastHistY = history.length > 0 ? getY(history[lastHistIdx][1]) : height;

    const rfPoints = forecast.map((d, i) => `${getX(lastHistIdx + 1 + i)},${getY(d.rf_value)}`).join(' ');
    const lstmPoints = forecast.map((d, i) => `${getX(lastHistIdx + 1 + i)},${getY(d.lstm_value)}`).join(' ');
    
    const rfPath = `${lastHistX},${lastHistY} ${rfPoints}`;
    const lstmPath = `${lastHistX},${lastHistY} ${lstmPoints}`;

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
            {[0, 0.25, 0.5, 0.75, 1].map((pct, i) => (
                <line key={i} x1="0" y1={height * pct} x2={graphWidth} y2={height * pct} stroke="#eee" strokeDasharray="4" vectorEffect="non-scaling-stroke" />
            ))}
            
            {/* Divider */}
            <line x1={lastHistX} y1="0" x2={lastHistX} y2={height} stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4" vectorEffect="non-scaling-stroke" />
            <text x={lastHistX + 5} y="15" className="text-xs fill-slate-400 font-bold">NOW</text>

            <polyline fill="none" stroke="#64748b" strokeWidth="2" points={histPoints} vectorEffect="non-scaling-stroke" />
            {showRF && <polyline fill="none" stroke="#0d9488" strokeWidth="2" points={rfPath} vectorEffect="non-scaling-stroke" />}
            {showLSTM && <polyline fill="none" stroke="#9333ea" strokeWidth="2" strokeDasharray="5,5" points={lstmPath} vectorEffect="non-scaling-stroke" />}

            {/* Interactive Points */}
            {history.map((d, i) => (
                <g key={`h-${i}`} className="group">
                    <rect x={getX(i) - 5} y="0" width="10" height={height} fill="transparent" className="cursor-pointer" >
                         <title>{`${new Date(d[0] * 1000).toLocaleTimeString('en-GB', {hour: '2-digit', minute: '2-digit', hour12: false})}\nHistory: ${Math.round(d[1])}`}</title>
                    </rect>
                    <circle cx={getX(i)} cy={getY(d[1])} r="3" fill="#64748b" className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
                </g>
            ))}

            {forecast.map((d, i) => {
                const x = getX(lastHistIdx + 1 + i);
                return (
                <g key={`f-${i}`} className="group">
                    <rect x={x - 5} y="0" width="10" height={height} fill="transparent" className="cursor-pointer" >
                        <title>{`${d.time}\nRF: ${d.rf_value}\nLSTM: ${d.lstm_value}`}</title>
                    </rect>
                    {showRF && <circle cx={x} cy={getY(d.rf_value)} r="4" fill="#0d9488" className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />}
                    {showLSTM && <circle cx={x} cy={getY(d.lstm_value)} r="4" fill="#9333ea" className="opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />}
                </g>
                )
            })}
            </svg>

            {/* X Axis */}
            <div className="absolute left-0 right-0 top-[310px] h-6 pointer-events-none">
                {allLabels.map((label, i) => {
                    if (i % 6 !== 0) return null;
                    const leftPct = (i / (totalPoints - 1)) * 100;
                    return (
                        <div key={i} className="absolute text-[10px] text-gray-400 transform -translate-x-1/2 -translate-y-full" style={{ left: `${leftPct}%` }}>
                            {label}
                        </div>
                    );
                })}
            </div>
        </div>
        
        <div className="flex justify-between mt-4 text-sm text-slate-500 font-mono">
            <span>-24 Hours</span>
            <span>Now</span>
            <span>+24 Hours</span>
        </div>
      </div>
    );
  };

  return (
    <div className="p-8 bg-teal-50 min-h-screen">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
        
        {/* Left Side: Back Button & Selectors */}
        <div className="flex items-center gap-4">
          <Link to={`/inspector/${probeId}`} className="p-2 bg-white rounded-full shadow-sm text-teal-600 hover:text-teal-800 transition">
              <ArrowLeft size={24} />
          </Link>
          <div>
              <h1 className="text-2xl font-bold text-teal-900 mb-1">Traffic Forecast</h1>
              
              <div className="flex flex-col sm:flex-row sm:items-center text-sm text-teal-600 gap-2">
                <span className="mr-1 hidden sm:inline">Probe:</span>
                
                <div className="flex gap-2">
                    {/* Department Selector */}
                    <div className="relative inline-block">
                      <select 
                        value={selectedDept} 
                        onChange={handleDeptChange}
                        className="appearance-none bg-teal-50 border border-teal-200 hover:border-teal-400 text-teal-800 font-medium py-1 pl-3 pr-8 rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer w-32 md:w-auto"
                      >
                        {departments.map(d => (
                          <option key={d} value={d}>{d}</option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-teal-500">
                        <Layers size={14} />
                      </div>
                    </div>

                    {/* Probe Selector */}
                    <div className="relative inline-block">
                      <select 
                        value={probeId} 
                        onChange={handleProbeChange}
                        className="appearance-none bg-white border border-teal-200 hover:border-teal-400 text-teal-800 font-bold py-1 pl-3 pr-8 rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer w-48 md:w-auto"
                      >
                        {filteredProbes.map(p => (
                          <option key={p.id} value={p.name}>{p.name}</option>
                        ))}
                      </select>
                      <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-teal-600">
                        <ChevronDown size={14} />
                      </div>
                    </div>
                </div>
              </div>
          </div>
        </div>

        {/* Right Side: Model Toggles */}
        <div className="flex gap-3 text-sm">
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

      <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100 mb-8">
        
        {/* Metric Selectors */}
        <div className="flex flex-col md:flex-row gap-6 mb-6">
            
            {/* 1. Interface Selector */}
            <div className="flex bg-teal-50 p-1 rounded-lg self-start">
                <button
                  onClick={() => setSelectedInterface('lan')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-md transition-all ${
                    selectedInterface === 'lan' ? 'bg-white text-teal-700 shadow-sm' : 'text-teal-400 hover:text-teal-600'
                  }`}
                >
                  <Network size={16} /> LAN
                </button>
                <button
                  onClick={() => setSelectedInterface('wlan')}
                  className={`flex items-center gap-2 px-4 py-2 text-sm font-bold rounded-md transition-all ${
                    selectedInterface === 'wlan' ? 'bg-white text-purple-700 shadow-sm' : 'text-teal-400 hover:text-purple-600'
                  }`}
                >
                  <Globe size={16} /> WLAN
                </button>
            </div>

            {/* 2. Metric Type Selector */}
            <div className="flex bg-teal-50 p-1 rounded-lg flex-wrap">
                <button
                  onClick={() => setSelectedMetric('ext_down')}
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-bold rounded-md transition-all ${
                    selectedMetric === 'ext_down' ? 'bg-white text-emerald-600 shadow-sm' : 'text-teal-400'
                  }`}
                >
                  <ArrowDown size={14} /> Ext Download
                </button>
                <button
                  onClick={() => setSelectedMetric('ext_up')}
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-bold rounded-md transition-all ${
                    selectedMetric === 'ext_up' ? 'bg-white text-blue-600 shadow-sm' : 'text-teal-400'
                  }`}
                >
                  <ArrowUpIcon size={14} /> Ext Upload
                </button>
                <div className="w-px h-6 bg-teal-200 mx-1 self-center"></div>
                <button
                  onClick={() => setSelectedMetric('int_down')}
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-bold rounded-md transition-all ${
                    selectedMetric === 'int_down' ? 'bg-white text-emerald-600 shadow-sm' : 'text-teal-400'
                  }`}
                >
                  <ArrowDown size={14} /> Int Download
                </button>
                <button
                  onClick={() => setSelectedMetric('int_up')}
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-bold rounded-md transition-all ${
                    selectedMetric === 'int_up' ? 'bg-white text-blue-600 shadow-sm' : 'text-teal-400'
                  }`}
                >
                  <ArrowUpIcon size={14} /> Int Upload
                </button>
                <div className="w-px h-6 bg-teal-200 mx-1 self-center"></div>
                <button
                  onClick={() => setSelectedMetric('ping_ext')}
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-bold rounded-md transition-all ${
                    selectedMetric === 'ping_ext' ? 'bg-white text-orange-600 shadow-sm' : 'text-teal-400'
                  }`}
                >
                  <Activity size={14} /> Ping (Ext)
                </button>
                <button
                  onClick={() => setSelectedMetric('ping_int')}
                  className={`flex items-center gap-2 px-3 py-2 text-xs font-bold rounded-md transition-all ${
                    selectedMetric === 'ping_int' ? 'bg-white text-orange-600 shadow-sm' : 'text-teal-400'
                  }`}
                >
                  <Activity size={14} /> Ping (Int)
                </button>
            </div>
        </div>

        <ForecastGraph history={activeData.history} forecast={activeData.forecast} max={activeData.max} />
      </div>

      {/* Prediction Summary */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100">
            <h3 className="font-bold text-teal-800 mb-4">Predicted Peak</h3>
            {(() => {
                const maxLoad = Math.max(...activeData.forecast.map(d => d.rf_value), ...activeData.forecast.map(d => d.lstm_value));
                return (
                    <div>
                        <div className="text-4xl font-bold text-teal-600">{maxLoad} {selectedMetric.includes('ping') ? 'ms' : 'Mbps'}</div>
                        <div className="text-sm text-slate-500">Expected in next 24h</div>
                    </div>
                );
            })()}
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100">
            <h3 className="font-bold text-teal-800 mb-4">Metric Details</h3>
            <div className="text-sm text-slate-600">
                <p><strong>Interface:</strong> {selectedInterface.toUpperCase()}</p>
                <p><strong>Metric:</strong> {selectedMetric.replace('_', ' ').toUpperCase()}</p>
                <p className="mt-2 text-xs text-slate-400">
                    This forecast uses historical patterns to predict future network behavior.
                    Divergence between models may indicate unusual activity or changing patterns.
                </p>
            </div>
        </div>
      </div>
    </div>
  );
};

export default ForecastDetails;