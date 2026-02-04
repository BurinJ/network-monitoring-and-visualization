import React, { useEffect, useState } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { ChevronDown, ArrowDown, ArrowUp, Globe, Network, Activity, AlertTriangle, CheckCircle, Eye, EyeOff, Layers, TrendingUp } from 'lucide-react';
import { API_BASE_URL } from '../config'; 

// --- GENERIC TREND GRAPH COMPONENT ---
const TrendGraph = ({ 
  historyDown, historyUp, 
  avgDown, avgUp, 
  color = 'teal', 
  height = 80,
  title = "Trend",
  unit = "Mbps",
  labelDown = "DL", 
  labelUp = "UL",   
  tooltipDown = "Download", 
  tooltipUp = "Upload",
  upperThreshold = null, 
  lowerThreshold = null,
  anomalies = [] // List of timestamps (seconds)
}) => {
  const [showDown, setShowDown] = useState(true);
  const [showUp, setShowUp] = useState(true);

  if (!historyDown || !Array.isArray(historyDown) || historyDown.length === 0) {
    return <div className={`h-[${height}px] flex items-center justify-center text-xs text-gray-300`}>No History Data</div>;
  }

  // Determine active datasets for scaling
  let activeVals = [];
  if (showDown) activeVals = [...activeVals, ...historyDown.map(d => d[1])];
  if (showUp && historyUp && Array.isArray(historyUp)) activeVals = [...activeVals, ...historyUp.map(d => d[1])];
  if (upperThreshold) activeVals.push(upperThreshold);

  const maxVal = activeVals.length > 0 ? Math.max(...activeVals) * 1.1 : 100;
  const minVal = 0;
  const range = maxVal - minVal || 1;

  const formatTime = (ts) => new Date(ts * 1000).toLocaleTimeString('en-GB', { day: 'numeric', month: 'numeric', year: '2-digit', hour: '2-digit', minute: '2-digit' });

  const getX = (i, len) => (i / (len - 1)) * 300;
  const getY = (val) => height - ((val - minVal) / range) * height;

  const generatePoints = (data) => {
    if (!data || !Array.isArray(data)) return '';
    return data.map((d, i) => `${getX(i, data.length)},${getY(d[1])}`).join(' ');
  };

  const pointsDown = showDown ? generatePoints(historyDown) : '';
  const pointsUp = showUp && historyUp ? generatePoints(historyUp) : '';
  
  const yUpper = upperThreshold ? getY(upperThreshold) : null;
  const yLower = lowerThreshold ? getY(lowerThreshold) : null;

  // Calculate anomaly X positions
  // We match anomaly timestamps to history timestamps to find index
  const anomalyLines = anomalies.map(ts => {
      // Find closest index in history (within 15 mins)
      const index = historyDown.findIndex(d => Math.abs(d[0] - ts) < 900); 
      if (index !== -1) {
          return getX(index, historyDown.length);
      }
      return null;
  }).filter(x => x !== null);

  return (
    <div className="flex-1 min-w-[200px] mt-2 mb-6 last:mb-0 pl-8 relative">
      <div className="flex justify-between items-center mb-2">
        <span className={`text-[10px] font-bold uppercase tracking-wider text-${color}-600`}>{title}</span>
        
        {/* Controls */}
        <div className="flex items-center gap-3">
          <div className="flex gap-2">
            <button onClick={() => setShowDown(!showDown)} className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded border transition-colors ${showDown ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-gray-50 text-gray-400 border-gray-200'}`} title={showDown ? `Hide ${tooltipDown}` : `Show ${tooltipDown}`}>
              {showDown ? <Eye size={10} /> : <EyeOff size={10} />} {labelDown}
            </button>
            {historyUp && Array.isArray(historyUp) && (
              <button onClick={() => setShowUp(!showUp)} className={`flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded border transition-colors ${showUp ? 'bg-blue-50 text-blue-700 border-blue-200' : 'bg-gray-50 text-gray-400 border-gray-200'}`} title={showUp ? `Hide ${tooltipUp}` : `Show ${tooltipUp}`}>
                {showUp ? <Eye size={10} /> : <EyeOff size={10} />} {labelUp}
              </button>
            )}
          </div>
          <span className="text-[10px] text-gray-400 border-l pl-2 border-gray-200">Max: {Math.round(maxVal)} {unit}</span>
        </div>
      </div>

      <div className="relative" style={{ height: `${height}px` }}>
        <div className="absolute left-[-32px] top-0 bottom-0 w-8 flex flex-col justify-between text-[9px] text-gray-400 font-mono pointer-events-none pr-1 text-right z-10">
          <span>{Math.round(maxVal)}</span>
          <span>{Math.round(maxVal / 2)}</span>
          <span>0</span>
        </div>

        <div className="relative border-b border-l border-slate-100 bg-slate-50/30 rounded-sm overflow-hidden h-full w-full">
          <svg viewBox={`0 0 300 ${height}`} className="w-full h-full" preserveAspectRatio="none">
            
            {/* ANOMALY LINES */}
            {anomalyLines.map((x, i) => (
               <g key={`anom-${i}`}>
                   <line x1={x} y1="0" x2={x} y2={height} stroke="#ef4444" strokeWidth="1" strokeDasharray="3,2" opacity="0.8" vectorEffect="non-scaling-stroke" />
                   {/* Marker on top */}
                   <path d={`M ${x} 0 L ${x-3} -4 L ${x+3} -4 Z`} fill="#ef4444" />
               </g>
            ))}

            {upperThreshold && <line x1="0" y1={yUpper} x2="300" y2={yUpper} stroke="#ef4444" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" vectorEffect="non-scaling-stroke" />}
            {lowerThreshold && <line x1="0" y1={yLower} x2="300" y2={yLower} stroke="#ef4444" strokeWidth="1" strokeDasharray="3,3" opacity="0.6" vectorEffect="non-scaling-stroke" />}

            {showDown && (
              <>
                <polyline fill="none" stroke="#10b981" strokeWidth="1.5" points={pointsDown} vectorEffect="non-scaling-stroke" />
                <polygon fill="#10b981" fillOpacity="0.1" points={`0,${height} ${pointsDown} 300,${height}`} />
                {historyDown.map((d, i) => {
                  const x = getX(i, historyDown.length);
                  const y = getY(d[1]);
                  // Highlight anomalies
                  const isAnom = anomalyLines.includes(x);
                  return (
                    <circle key={`d-${i}`} cx={x} cy={y} r={isAnom ? 4 : 2.5} fill={isAnom ? '#ef4444' : '#10b981'} className="hover:r-5 transition-all cursor-pointer opacity-0 hover:opacity-100">
                      <title>{`${formatTime(d[0])} - ${historyUp ? `${tooltipDown}: ` : ''}${Math.round(d[1])} ${unit}${isAnom ? ' (AI Anomaly)' : ''}`}</title>
                    </circle>
                  );
                })}
              </>
            )}

            {showUp && historyUp && (
              <>
                <polyline fill="none" stroke="#3b82f6" strokeWidth="1.5" points={pointsUp} vectorEffect="non-scaling-stroke" />
                {historyUp.map((d, i) => {
                  const x = getX(i, historyUp.length);
                  const y = getY(d[1]);
                  return (
                    <circle key={`u-${i}`} cx={x} cy={y} r="2.5" fill="#3b82f6" className="hover:r-5 transition-all cursor-pointer opacity-0 hover:opacity-100">
                      <title>{`${formatTime(d[0])} - ${tooltipUp}: ${Math.round(d[1])} ${unit}`}</title>
                    </circle>
                  );
                })}
              </>
            )}
          </svg>
        </div>
      </div>
      <div className="flex justify-between mt-1 text-[9px] text-slate-400 font-mono pl-1">
        <span>{showDown ? `Avg ${historyUp ? labelDown : ''}: ${Math.round(avgDown)} ${unit}` : ''}</span>
        
        {/* Footer Legend */}
        <div className="flex gap-3">
             {(upperThreshold || lowerThreshold) && <span className="text-red-400">Limit: {upperThreshold || lowerThreshold}</span>}
             {anomalyLines.length > 0 && (
                 <span className="text-red-500 font-bold flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div> 
                    {anomalyLines.length} Events Detected
                 </span>
             )}
        </div>

        {historyUp && <span>{showUp ? `Avg ${labelUp}: ${Math.round(avgUp)} ${unit}` : ''}</span>}
      </div>
    </div>
  );
};

// --- UPDATED DIAGNOSIS CARD ---
const DiagnosisCard = ({ diag }) => {
  const [isOpen, setIsOpen] = useState(false);
  const hasDetails = diag.causes && diag.causes.length > 0;

  return (
    <div className={`border-l-4 p-4 rounded-r-xl shadow-sm bg-white mb-4 ${
        diag.status === 'Healthy' ? 'border-emerald-500' : 
        diag.status === 'Warning' ? 'border-orange-500' : 'border-red-500'
    }`}>
       <div className="flex items-start gap-4">
          <div className={`mt-1 ${diag.status === 'Healthy' ? 'text-emerald-500' : 'text-red-500'}`}>
              {diag.status === 'Healthy' ? <CheckCircle size={24} /> : <AlertTriangle size={24} />}
          </div>
          <div className="flex-1">
              <div className="flex justify-between items-start">
                 <div>
                    <h3 className={`font-bold mb-1 ${diag.status === 'Healthy' ? 'text-emerald-800' : 'text-red-800'}`}>
                       {diag.status === 'Healthy' ? '✔' : '⚠️'} {diag.title}
                    </h3>
                    <p className="text-sm opacity-80 text-slate-600">{diag.desc}</p>
                 </div>
                 {hasDetails && (
                   <button onClick={() => setIsOpen(!isOpen)} className="text-slate-400 hover:text-slate-600 p-1">
                     <ChevronDown size={18} className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                   </button>
                 )}
              </div>
              
              {isOpen && hasDetails && (
                 <div className="mt-3 pt-3 border-t border-slate-100">
                    <div className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Possible Root Causes</div>
                    {diag.causes.map((c, i) => (
                       <div key={i} className="mb-2 last:mb-0 p-2 bg-slate-50 rounded border border-slate-100">
                          <div className="font-bold text-xs text-slate-700 flex items-center gap-1.5 mb-1">
                            <div className="w-1.5 h-1.5 rounded-full bg-red-400"></div>
                            {c.title}
                          </div>
                          <div className="text-xs text-slate-500 pl-3">{c.detail}</div>
                       </div>
                    ))}
                 </div>
              )}
           </div>
       </div>
    </div>
  )
};

const Inspector = () => {
  const { probeId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [allProbes, setAllProbes] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [selectedDept, setSelectedDept] = useState('All');
  const [timeRange, setTimeRange] = useState('24h');
  
  const [thresholds, setThresholds] = useState({
    lan_ping: 100, wlan_ping: 200, dns: 100
  });

  // 1. Fetch Probe List and Thresholds
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    
    fetch(`${baseUrl}/settings/probes`)
      .then(res => res.json())
      .then(data => {
        setAllProbes(data);
        const depts = Array.from(new Set(data.map(p => p.department || 'Undefined')));
        setDepartments(['All', ...depts.sort()]);
        const current = data.find(p => p.name === probeId);
        if (current) setSelectedDept(current.department || 'Undefined');
      }).catch(console.error);

    fetch(`${baseUrl}/settings/thresholds`)
      .then(res => res.json())
      .then(data => {
        if(data) setThresholds({
            lan_ping: parseFloat(data.lan_ping_threshold) || 100,
            wlan_ping: parseFloat(data.wlan_ping_threshold) || 200,
            dns: parseFloat(data.dns_threshold) || 100
        });
      }).catch(console.error);
  }, []);

  // Sync selected dept
  useEffect(() => {
    if (allProbes.length > 0) {
        const current = allProbes.find(p => p.name === probeId);
        if (current) setSelectedDept(current.department || 'Undefined');
    }
  }, [probeId, allProbes]);

  // 2. Fetch Data
  useEffect(() => {
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    setData(null);
    fetch(`${baseUrl}/inspector/${probeId}?duration=${timeRange}`)
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, [probeId, timeRange]);

  const handleProbeChange = (e) => navigate(`/inspector/${e.target.value}`);
  const handleDeptChange = (e) => {
    const newDept = e.target.value;
    setSelectedDept(newDept);
    const firstProbe = allProbes.find(p => (newDept === 'All' || (p.department || 'Undefined') === newDept));
    if (firstProbe && firstProbe.name !== probeId) navigate(`/inspector/${firstProbe.name}`);
  };
  const handleTimeRangeChange = (range) => setTimeRange(range);

  const filteredProbes = allProbes.filter(p => selectedDept === 'All' || (p.department || 'Undefined') === selectedDept)
    .sort((a, b) => a.name.localeCompare(b.name));

  if (!data) return <div className="p-10 text-teal-600 font-medium animate-pulse">Loading Probe Data...</div>;

  const { ai_diagnoses = [], metrics, has_wlan } = data;
  const anomalies = metrics.anomalies || [];
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
            <div className={`h-full rounded-full ${barColor} transition-all duration-1000 ease-out`} style={{ width: `${percent}%` }} />
          </div>
        </div>
      </div>
    );
  };

  const SpeedSection = ({ title, icon: TitleIcon, external, internal, history, average, color, speedCap, anomalies }) => (
    <div className="bg-white border border-teal-100 rounded-lg p-4 shadow-sm">
      <h4 className="font-bold text-teal-600 text-xs uppercase mb-4 flex items-center gap-2 border-b border-teal-50 pb-2">
        <TitleIcon size={14} /> {title} Speed <span className="text-gray-400 text-[10px] ml-auto">Cap: {speedCap} Mbps</span>
      </h4>
      <div className="mb-6">
        <span className="text-[10px] uppercase font-bold text-slate-400 mb-2 block tracking-wider">External Internet</span>
        <SpeedBar label="Download" value={external.down} max={speedCap} barColor="bg-emerald-500" iconColor="emerald" Icon={ArrowDown} />
        <SpeedBar label="Upload" value={external.up} max={speedCap} barColor="bg-blue-500" iconColor="blue" Icon={ArrowUp} />
        <TrendGraph title={`External Trend (${timeRange})`} historyDown={history.external.down} historyUp={history.external.up} avgDown={average.external.down} avgUp={average.external.up} color={color} height={120} lowerThreshold={speedCap * 0.05} anomalies={anomalies} />
      </div>
      <div className="pt-4 border-t border-dashed border-teal-50">
        <span className="text-[10px] uppercase font-bold text-slate-400 mb-2 block tracking-wider">Internal LAN</span>
        <SpeedBar label="Download" value={internal.down} max={speedCap} barColor="bg-emerald-400" iconColor="emerald" Icon={ArrowDown} />
        <SpeedBar label="Upload" value={internal.up} max={speedCap} barColor="bg-blue-400" iconColor="blue" Icon={ArrowUp} />
        <TrendGraph title={`Internal Trend (${timeRange})`} historyDown={history.internal.down} historyUp={history.internal.up} avgDown={average.internal.down} avgUp={average.internal.up} color={color} height={120} lowerThreshold={speedCap * 0.05} anomalies={anomalies} />
      </div>
    </div>
  );

  const LatencySection = ({ history, average, color, pingThreshold, anomalies }) => (
    <div className="bg-white border border-teal-100 rounded-lg p-4 shadow-sm">
      <h4 className="font-bold text-teal-600 text-xs uppercase mb-4 flex items-center gap-2 border-b border-teal-50 pb-2">
        <Activity size={14} /> Latency Trends
      </h4>
      <TrendGraph title={`Ping History (${timeRange})`} historyDown={history.external} historyUp={history.internal} avgDown={average.external} avgUp={average.internal} color={color} height={100} unit="ms" labelDown="Ext" labelUp="Int" tooltipDown="External Ping" tooltipUp="Internal Ping" upperThreshold={pingThreshold} anomalies={anomalies} />
    </div>
  );

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-8 gap-4">
         <div>
            <h1 className="text-2xl font-bold text-teal-900 mb-1">Probe Inspection</h1>
            <div className="flex flex-col sm:flex-row sm:items-center text-sm text-teal-600 gap-2">
              <span className="mr-1 hidden sm:inline">Inspect:</span>
              <div className="flex gap-2">
                  <div className="relative inline-block">
                    <select value={selectedDept} onChange={handleDeptChange} className="appearance-none bg-teal-50 border border-teal-200 hover:border-teal-400 text-teal-800 font-medium py-1 pl-3 pr-8 rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer w-32 md:w-auto">
                      {departments.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-teal-500"><Layers size={14} /></div>
                  </div>
                  <div className="relative inline-block">
                    <select value={probeId} onChange={handleProbeChange} className="appearance-none bg-white border border-teal-200 hover:border-teal-400 text-teal-800 font-bold py-1 pl-3 pr-8 rounded shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 cursor-pointer w-48 md:w-auto">
                      {filteredProbes.map(p => <option key={p.id} value={p.name}>{p.name}</option>)}
                    </select>
                    <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-teal-600"><ChevronDown size={14} /></div>
                  </div>
              </div>
            </div>
         </div>
         
         <div className="flex gap-2">
            {['1h', '24h', '1w'].map(range => (
              <button key={range} onClick={() => handleTimeRangeChange(range)} className={`px-3 py-1 text-xs font-bold rounded border transition-colors ${timeRange === range ? 'bg-teal-600 text-white border-teal-600' : 'bg-white text-teal-600 border-teal-200 hover:bg-teal-50'}`}>{range.toUpperCase()}</button>
            ))}
            <Link to="/admin" className="ml-2 bg-white border border-teal-200 shadow-sm px-4 py-1.5 rounded text-sm text-teal-700 hover:bg-teal-50 transition-colors flex items-center">Back to Issues</Link>
         </div>
      </div>

      <div className="grid grid-cols-1 gap-4 mb-8">
        {ai_diagnoses.map((diag, index) => (
           <DiagnosisCard key={index} diag={diag} />
        ))}
      </div>

      <div className={`grid grid-cols-1 ${showWlan ? 'lg:grid-cols-2' : ''} gap-8`}>
        <div className="bg-teal-100/50 p-6 rounded-2xl border border-teal-200 flex flex-col gap-6">
          <h3 className="text-lg font-bold text-teal-800 flex items-center gap-2"><span className="bg-white p-1 rounded shadow-sm">🌐</span> Ethernet (LAN) Status</h3>
          <IpDisplay v4={metrics.lan.ipv4} v6={metrics.lan.ipv6} />
          <div className="grid grid-cols-2 gap-4">
            <MetricCard label="Ping Latency" value={metrics.lan.ping} unit="ms" ideal={thresholds.lan_ping} />
            <MetricCard label="DNS Response" value={metrics.lan.dns} unit="ms" ideal={thresholds.dns} />
          </div>
          <LatencySection history={metrics.lan.history.ping} average={metrics.lan.average.ping} color="emerald" pingThreshold={thresholds.lan_ping} anomalies={anomalies} />
          <SpeedSection title="Wired" icon={Network} external={metrics.lan.speed.external} internal={metrics.lan.speed.internal} history={metrics.lan.history} average={metrics.lan.average} color="emerald" speedCap={metrics.lan.speed_cap} anomalies={anomalies} />
        </div>

        {showWlan && (
          <div className="bg-purple-50 p-6 rounded-2xl border border-purple-100 flex flex-col gap-6">
            <h3 className="text-lg font-bold text-purple-800 flex items-center gap-2"><span className="bg-white p-1 rounded shadow-sm">📶</span> Wi-Fi (WLAN) Status</h3>
            <IpDisplay v4={metrics.wlan.ipv4} v6={metrics.wlan.ipv6} />
            <div className="grid grid-cols-2 gap-4">
              <MetricCard label="Ping Latency" value={metrics.wlan.ping} unit="ms" ideal={thresholds.wlan_ping} />
              <MetricCard label="DNS Response" value={metrics.wlan.dns} unit="ms" ideal={thresholds.dns} />
            </div>
            <LatencySection history={metrics.wlan.history.ping} average={metrics.wlan.average.ping} color="purple" pingThreshold={thresholds.wlan_ping} anomalies={anomalies} />
            <SpeedSection title="Wireless" icon={Globe} external={metrics.wlan.speed.external} internal={metrics.wlan.speed.internal} history={metrics.wlan.history} average={metrics.wlan.average} color="purple" speedCap={metrics.wlan.speed_cap} anomalies={anomalies} />
          </div>
        )}
      </div>
    </div>
  );
};

export default Inspector;