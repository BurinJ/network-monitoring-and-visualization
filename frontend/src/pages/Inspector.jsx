import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ChevronDown, ArrowDown, ArrowUp, Globe, Network, Activity, AlertTriangle, CheckCircle } from 'lucide-react';
import { API_BASE_URL } from '../config'; 

// ... (TrendGraph component remains the same) ...
// --- GENERIC TREND GRAPH COMPONENT (Renamed from SpeedHistoryGraph) ---
const TrendGraph = ({ 
  historyDown, historyUp, 
  avgDown, avgUp, 
  color = 'teal', 
  height = 80,
  title = "Trend",
  unit = "Mbps"
}) => {
  if (!historyDown || historyDown.length === 0) return <div className={`h-[${height}px] flex items-center justify-center text-xs text-gray-300`}>No History Data</div>;

  const valsDown = historyDown.map(d => d[1]);
  const valsUp = historyUp ? historyUp.map(d => d[1]) : [];
  const allVals = [...valsDown, ...valsUp];
  
  const maxVal = Math.max(...allVals, 10);
  const minVal = 0;
  const range = maxVal - minVal || 1;

  const generatePoints = (data) => {
    return data.map((d, i) => {
      const x = (i / (data.length - 1)) * 300;
      const y = height - ((d[1] - minVal) / range) * height;
      return `${x},${y}`;
    }).join(' ');
  };

  const pointsDown = generatePoints(historyDown);
  const pointsUp = historyUp ? generatePoints(historyUp) : '';

  const getAvgY = (val) => val ? height - ((val - minVal) / range) * height : 0;
  const yAvgDown = getAvgY(avgDown);
  const yAvgUp = getAvgY(avgUp);

  return (
    <div className="flex-1 min-w-[200px] mt-2 mb-6 last:mb-0">
      <div className="flex justify-between items-center mb-2">
        <span className={`text-[10px] font-bold uppercase tracking-wider text-${color}-600`}>{title}</span>
        <span className="text-[10px] text-gray-400">Max: {Math.round(maxVal)} {unit}</span>
      </div>

      <div className="relative border-b border-l border-slate-100 bg-slate-50/30 rounded-sm overflow-hidden" style={{ height: `${height}px` }}>
        <svg viewBox={`0 0 300 ${height}`} className="w-full h-full" preserveAspectRatio="none">
          
          {/* Line 1 (Download or Ping) */}
          <polyline 
            fill="none" 
            stroke="#10b981" // emerald-500
            strokeWidth="1.5" 
            points={pointsDown} 
            vectorEffect="non-scaling-stroke"
          />
          {avgDown > 0 && (
            <line 
              x1="0" y1={yAvgDown} x2="300" y2={yAvgDown} 
              stroke="#10b981" 
              strokeWidth="1" 
              strokeDasharray="4,2" 
              opacity="0.6"
              vectorEffect="non-scaling-stroke"
            />
          )}

          {/* Line 2 (Upload - Optional) */}
          {historyUp && (
            <>
              <polyline 
                fill="none" 
                stroke="#3b82f6" // blue-500
                strokeWidth="1.5" 
                points={pointsUp} 
                vectorEffect="non-scaling-stroke"
              />
              {avgUp > 0 && (
                <line 
                  x1="0" y1={yAvgUp} x2="300" y2={yAvgUp} 
                  stroke="#3b82f6" 
                  strokeWidth="1" 
                  strokeDasharray="4,2" 
                  opacity="0.6"
                  vectorEffect="non-scaling-stroke"
                />
              )}
            </>
          )}
        </svg>
      </div>
      
      <div className="flex justify-between mt-1 text-[9px] text-slate-400 font-mono">
        <span>Avg {historyUp ? 'DL' : ''}: {Math.round(avgDown)} {unit}</span>
        {historyUp && <span>Avg UL: {Math.round(avgUp)} {unit}</span>}
      </div>
    </div>
  );
};

const Inspector = () => {
  const { probeId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [probes, setProbes] = useState([]);
  const [timeRange, setTimeRange] = useState('24h');

  // 1. Fetch Probe List
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/probes`)
      .then(res => res.json())
      .then(setProbes)
      .catch(console.error);
  }, []);

  // 2. Fetch Data
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    setData(null);
    fetch(`${baseUrl}/inspector/${probeId}?duration=${timeRange}`)
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, [probeId, timeRange]);

  const handleProbeChange = (e) => {
    navigate(`/inspector/${e.target.value}`);
  };

  const handleTimeRangeChange = (range) => {
    setTimeRange(range);
  };

  if (!data) return <div className="p-10 text-teal-600 font-medium animate-pulse">Loading Probe Data...</div>;

  // Destructure with default empty list for diagnoses
  const { ai_diagnoses = [], metrics, has_wlan } = data;
  const showWlan = has_wlan !== undefined ? has_wlan : (metrics.wlan.ping > 0 || metrics.wlan.dns > 0);

  const IpDisplay = ({ v4, v6 }) => (
    <div className="bg-white border border-teal-100 rounded-lg p-3 text-xs font-mono text-slate-600 mb-4 break-all shadow-sm">
      {v4 && <div><span className="text-teal-400 font-bold select-none">IPv4: </span>{v4}</div>}
      {v6 && <div className="mt-1"><span className="text-teal-400 font-bold select-none">IPv6: </span>{v6}</div>}
      {!v4 && !v6 && <div className="italic text-gray-400">No IP Address Detected</div>}
    </div>
  );

  const MetricCard = ({ label, value, unit, ideal }) => (
    <div className="bg-white border border-teal-100 rounded-lg p-4 flex flex-col items-center justify-center shadow-sm h-32">
      <h4 className="font-bold text-teal-500 text-xs uppercase mb-2 tracking-wider">{label}</h4>
      <div className={`text-3xl font-mono ${value > ideal ? 'text-orange-500' : 'text-teal-800'}`}>
        {value} <span className="text-sm text-teal-400 font-sans">{unit}</span>
      </div>
      <div className="text-[10px] text-teal-400 mt-1 bg-teal-50 px-2 py-0.5 rounded">
        Target: &lt;{ideal}{unit}
      </div>
    </div>
  );

  const SpeedBar = ({ label, value, max = 1000, barColor, iconColor, Icon }) => {
    const percent = Math.min((value / max) * 100, 100);
    return (
      <div className="flex items-center gap-3 mb-3 last:mb-0">
        <div className={`p-1.5 rounded-full bg-${iconColor}-50 text-${iconColor}-600`}>
          <Icon size={14} />
        </div>
        <div className="flex-1">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-500 font-medium">{label}</span>
            <span className="font-mono font-bold text-slate-700">{value} Mbps</span>
          </div>
          <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
            <div 
              className={`h-full rounded-full ${barColor} transition-all duration-1000 ease-out`}
              style={{ width: `${percent}%` }}
            />
          </div>
        </div>
      </div>
    );
  };

  const SpeedSection = ({ title, icon: TitleIcon, external, internal, history, average, color }) => (
    <div className="bg-white border border-teal-100 rounded-lg p-4 shadow-sm">
      <h4 className="font-bold text-teal-600 text-xs uppercase mb-4 flex items-center gap-2 border-b border-teal-50 pb-2">
        <TitleIcon size={14} /> {title} Speed
      </h4>
      
      <div className="mb-6">
        <span className="text-[10px] uppercase font-bold text-slate-400 mb-2 block tracking-wider">External Internet</span>
        <SpeedBar label="Download" value={external.down} barColor="bg-emerald-500" iconColor="emerald" Icon={ArrowDown} />
        <SpeedBar label="Upload" value={external.up} barColor="bg-blue-500" iconColor="blue" Icon={ArrowUp} />
        
        <TrendGraph 
          title={`External Trend (${timeRange})`}
          historyDown={history.external.down} 
          historyUp={history.external.up} 
          avgDown={average.external.down} 
          avgUp={average.external.up} 
          color={color}
          height={120} 
        />
      </div>

      <div className="pt-4 border-t border-dashed border-teal-50">
        <span className="text-[10px] uppercase font-bold text-slate-400 mb-2 block tracking-wider">Internal LAN</span>
        <SpeedBar label="Download" value={internal.down} barColor="bg-emerald-400" iconColor="emerald" Icon={ArrowDown} />
        <SpeedBar label="Upload" value={internal.up} barColor="bg-blue-400" iconColor="blue" Icon={ArrowUp} />

        <TrendGraph 
          title={`Internal Trend (${timeRange})`}
          historyDown={history.internal.down} 
          historyUp={history.internal.up} 
          avgDown={average.internal.down} 
          avgUp={average.internal.up} 
          color={color} 
          height={120} 
        />
      </div>
    </div>
  );

  const LatencySection = ({ history, average, color }) => (
    <div className="bg-white border border-teal-100 rounded-lg p-4 shadow-sm">
      <h4 className="font-bold text-teal-600 text-xs uppercase mb-4 flex items-center gap-2 border-b border-teal-50 pb-2">
        <Activity size={14} /> Latency Trends
      </h4>
      <TrendGraph 
        title={`Ping History (${timeRange})`}
        historyDown={history} 
        avgDown={average}     
        color={color}
        height={80} 
        unit="ms"
      />
    </div>
  );

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      
      {/* Header Row */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
         <div>
            <h1 className="text-2xl font-bold text-teal-900 mb-1">Probe Inspection</h1>
            <div className="flex items-center text-sm text-teal-600">
              <span className="mr-2">Inspecting:</span>
              <div className="relative inline-block">
                <select 
                  value={probeId} 
                  onChange={handleProbeChange}
                  className="appearance-none bg-white border border-teal-200 hover:border-teal-400 text-teal-800 font-bold py-1 pl-3 pr-8 rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer"
                >
                  {probes.map(p => <option key={p} value={p}>{p}</option>)}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-teal-600">
                  <ChevronDown size={14} />
                </div>
              </div>
            </div>
         </div>
         
         <div className="flex gap-2">
            {['1h', '24h', '1w'].map(range => (
              <button
                key={range}
                onClick={() => handleTimeRangeChange(range)}
                className={`px-3 py-1 text-xs font-bold rounded border transition-colors
                  ${timeRange === range 
                    ? 'bg-teal-600 text-white border-teal-600' 
                    : 'bg-white text-teal-600 border-teal-200 hover:bg-teal-50'}`}
              >
                {range.toUpperCase()}
              </button>
            ))}
            <Link to="/admin" className="ml-4 bg-white border border-teal-200 shadow-sm px-4 py-1.5 rounded text-sm text-teal-700 hover:bg-teal-50 transition-colors flex items-center">
              Back to Issues
            </Link>
         </div>
      </div>

      {/* MULTIPLE AI DIAGNOSES LIST */}
      <div className="grid grid-cols-1 gap-4 mb-8">
        {ai_diagnoses.map((diag, index) => (
          <div key={index} className={`border-l-4 p-6 rounded-r-xl shadow-sm bg-white flex items-start gap-4 ${
            diag.status === 'Healthy' ? 'border-emerald-500' : 
            diag.status === 'Warning' ? 'border-orange-500' : 'border-red-500'
          }`}>
            <div className={`mt-1 ${diag.status === 'Healthy' ? 'text-emerald-500' : 'text-red-500'}`}>
              {diag.status === 'Healthy' ? <CheckCircle size={24} /> : <AlertTriangle size={24} />}
            </div>
            <div>
              <h3 className={`font-bold mb-1 ${diag.status === 'Healthy' ? 'text-emerald-800' : 'text-red-800'}`}>
                 {diag.status === 'Healthy' ? '✔' : '⚠️'} {diag.title}
              </h3>
              <p className="text-sm opacity-80 text-slate-600">{diag.desc}</p>
            </div>
          </div>
        ))}
      </div>

      <div className={`grid grid-cols-1 ${showWlan ? 'lg:grid-cols-2' : ''} gap-8`}>
        {/* LAN COLUMN */}
        <div className="bg-teal-100/50 p-6 rounded-2xl border border-teal-200 flex flex-col gap-6">
          <h3 className="text-lg font-bold text-teal-800 flex items-center gap-2">
            <span className="bg-white p-1 rounded shadow-sm">🌐</span> Ethernet (LAN) Status
          </h3>
          
          <IpDisplay v4={metrics.lan.ipv4} v6={metrics.lan.ipv6} />

          <div className="grid grid-cols-2 gap-4">
            <MetricCard label="Ping Latency" value={metrics.lan.ping} unit="ms" ideal={50} />
            <MetricCard label="DNS Response" value={metrics.lan.dns} unit="ms" ideal={50} />
          </div>

          <LatencySection 
            history={metrics.lan.history.ping} 
            average={metrics.lan.average.ping} 
            color="emerald" 
          />

          <SpeedSection 
            title="Wired" 
            icon={Network}
            external={metrics.lan.speed.external}
            internal={metrics.lan.speed.internal}
            history={metrics.lan.history}
            average={metrics.lan.average}
            color="emerald"
          />
        </div>

        {/* WLAN COLUMN */}
        {showWlan && (
          <div className="bg-purple-50 p-6 rounded-2xl border border-purple-100 flex flex-col gap-6">
            <h3 className="text-lg font-bold text-purple-800 flex items-center gap-2">
              <span className="bg-white p-1 rounded shadow-sm">📶</span> Wi-Fi (WLAN) Status
            </h3>

            <IpDisplay v4={metrics.wlan.ipv4} v6={metrics.wlan.ipv6} />

            <div className="grid grid-cols-2 gap-4">
              <MetricCard label="Ping Latency" value={metrics.wlan.ping} unit="ms" ideal={100} />
              <MetricCard label="DNS Response" value={metrics.wlan.dns} unit="ms" ideal={100} />
            </div>

            <LatencySection 
              history={metrics.wlan.history.ping} 
              average={metrics.wlan.average.ping} 
              color="purple" 
            />

            <SpeedSection 
              title="Wireless" 
              icon={Globe}
              external={metrics.wlan.speed.external}
              internal={metrics.wlan.speed.internal}
              history={metrics.wlan.history}
              average={metrics.wlan.average}
              color="purple"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Inspector;