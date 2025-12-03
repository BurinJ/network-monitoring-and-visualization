import GrafanaPanel from '../components/GrafanaPanel';

const Overview = () => {
  return (
    <div className="p-6 bg-gray-50 min-h-screen ml-64">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Overview</h1>
      
      {/* Top Row: 3 Columns (Status, Dept Status, Performance) */}
      <div className="grid grid-cols-12 gap-6 mb-6">
        
        {/* Left: Probe Status (Donut Chart) */}
        <div className="col-span-3 h-80">
          {/* Use panelId of your Donut chart in Grafana */}
          <GrafanaPanel panelId={1} title="Probe Status" /> 
        </div>

        {/* Middle: Dept Status & Map */}
        <div className="col-span-5 flex flex-col gap-6">
           <div className="h-40">
             <GrafanaPanel panelId={2} title="Department Status" />
           </div>
           <div className="h-40 bg-gray-200 rounded-xl flex items-center justify-center">
             {/* Placeholder for Map - or use a Grafana Map Panel */}
             <span className="text-gray-500">Network Map Visualization</span>
           </div>
        </div>

        {/* Right: Performance Table */}
        <div className="col-span-4 bg-white rounded-xl shadow-sm border p-4">
          <h3 className="font-bold mb-4">WLAN External Speed</h3>
          <table className="w-full text-sm text-left">
            <thead className="bg-gray-100 text-gray-600">
              <tr><th className="p-2">Probe</th><th className="p-2">Down</th><th className="p-2">Up</th></tr>
            </thead>
            <tbody>
              {/* This data will eventually come from your API */}
              <tr className="border-b"><td className="p-2">Probe1</td><td>158</td><td>91.5</td></tr>
              <tr className="border-b"><td className="p-2">Probe2</td><td>128</td><td>313</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Bottom Row: Downtime Table */}
      <div className="bg-white p-6 rounded-xl shadow-sm border">
        <h3 className="font-bold mb-4">Downtime Reasons</h3>
        {/* Standard HTML Table mimicking Image 1 */}
        <div className="w-full h-32 bg-gray-50 border border-dashed border-gray-300 flex items-center justify-center">
             Table Data Placeholder
        </div>
      </div>
    </div>
  );
};

export default Overview;