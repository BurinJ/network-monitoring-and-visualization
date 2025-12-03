import GrafanaPanel from '../components/GrafanaPanel.jsx';

const ProbeDetails = () => {
  return (
    <div className="p-6 bg-gray-50 min-h-screen ml-64">
      
      {/* Header Section */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800">Probe Details</h1>
        <div className="flex gap-4 text-sm text-gray-500 mt-1">
          <span>Department: <strong>Dep1</strong></span>
          <span>Probe Name: <strong>Probe1</strong></span>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        
        {/* LEFT COLUMN: Info & AI Root Causes (30% Width) */}
        <div className="col-span-4 space-y-6">
          
          {/* Properties Card */}
          <div className="bg-white p-6 rounded-xl shadow-sm border">
            <h3 className="text-gray-500 uppercase text-xs font-bold mb-4">Properties</h3>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-500">IPv4:</span> 192.168.1.10</p>
              <p><span className="text-gray-500">Status:</span> <span className="text-red-500 font-bold">Downed</span></p>
              <p><span className="text-gray-500">Duration:</span> 2 days 4 hours</p>
            </div>
          </div>

          {/* AI Analysis Section - Matches Image 3 "Problems" & "Root Causes" */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-red-100">
            
            {/* Problems */}
            <div className="mb-6">
              <h3 className="text-orange-500 font-bold mb-2">Problems</h3>
              <ul className="list-disc pl-5 text-sm text-gray-600 space-y-2">
                <li>
                  <span className="font-semibold text-gray-800">Reason 1</span>
                  <p className="text-xs mt-1">Anomaly detected: Packet loss spiked to 80% at 14:00.</p>
                </li>
              </ul>
            </div>

            {/* Root Causes */}
            <div>
              <h3 className="text-green-600 font-bold mb-2">Possible Root Causes</h3>
              <ul className="list-disc pl-5 text-sm text-gray-600">
                <li>
                  <span className="font-semibold text-gray-800">Switch Overload</span>
                  <p className="text-xs mt-1">High correlation with CPU spike on Switch-Lib-01. Recommend rebooting.</p>
                </li>
              </ul>
            </div>

          </div>
        </div>

        {/* RIGHT COLUMN: Grafana Charts Grid (70% Width) */}
        <div className="col-span-8 grid grid-cols-2 gap-4">
           {/* These replicate the 4 charts in Image 3 */}
           <GrafanaPanel title="Lan Latency (ms)" panelId={3} />
           <GrafanaPanel title="Wifi Latency (ms)" panelId={4} />
           <GrafanaPanel title="Lan Internet Speed" panelId={5} />
           <GrafanaPanel title="Wifi Internet Speed" panelId={6} />
        </div>

      </div>
    </div>
  );
};

export default ProbeDetails;