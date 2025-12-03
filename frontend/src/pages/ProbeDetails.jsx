import React from 'react';

const ProbeDetails = () => {
  return (
    <div className="p-6 bg-gray-50 min-h-screen text-gray-800">
      
      {/* Header Section */}
      <div className="mb-6 border-b border-gray-200 pb-4">
        <h1 className="text-2xl font-bold text-slate-800">Probe Details</h1>
        <div className="flex gap-6 mt-3 text-sm">
          <div className="bg-white px-3 py-1 rounded border shadow-sm">
            <span className="text-gray-500 mr-2">Department:</span>
            <span className="font-bold">Engineering (Dep1)</span>
          </div>
          <div className="bg-white px-3 py-1 rounded border shadow-sm">
            <span className="text-gray-500 mr-2">Probe Name:</span>
            <span className="font-bold">Probe-Alpha-01</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-6">
        
        {/* LEFT COLUMN: Properties & AI Analysis (30% Width) */}
        <div className="col-span-4 space-y-6">
          
          {/* Properties Card */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
            <h3 className="text-gray-400 uppercase text-xs font-bold mb-4 tracking-wider">Properties</h3>
            <div className="space-y-3 text-sm">
              <div className="flex justify-between border-b border-gray-100 pb-2">
                <span className="text-gray-500">IPv4 Address</span>
                <span className="font-mono text-gray-700">192.168.10.45</span>
              </div>
              <div className="flex justify-between border-b border-gray-100 pb-2">
                 <span className="text-gray-500">Status</span>
                 <span className="bg-red-100 text-red-600 px-2 py-0.5 rounded text-xs font-bold uppercase">Downed</span>
              </div>
              <div className="flex justify-between">
                 <span className="text-gray-500">Duration</span>
                 <span>2 days 4 hours</span>
              </div>
            </div>
          </div>

          {/* AI Analysis Section (The "Smart" Part) */}
          <div className="bg-white p-6 rounded-xl shadow-sm border border-red-200 relative overflow-hidden">
            <div className="absolute top-0 left-0 w-1 h-full bg-red-400"></div>
            
            {/* Problems List */}
            <div className="mb-6">
              <h3 className="text-orange-500 font-bold mb-2 flex items-center gap-2">
                ⚠ Problems Detected
              </h3>
              <ul className="list-disc pl-5 text-sm text-gray-600 space-y-3">
                <li>
                  <span className="font-semibold text-gray-800">High Latency Spike</span>
                  <p className="text-xs mt-1 text-gray-500">
                    Latency exceeded 500ms threshold at 14:00. Flagged by Anomaly Detection.
                  </p>
                </li>
                <li>
                  <span className="font-semibold text-gray-800">Packet Loss &gt; 15%</span>
                  <p className="text-xs mt-1 text-gray-500">
                    Consistent packet drops observed on outbound traffic.
                  </p>
                </li>
              </ul>
            </div>

            {/* Root Causes */}
            <div className="pt-4 border-t border-gray-100">
              <h3 className="text-green-600 font-bold mb-2 flex items-center gap-2">
                ✔ Possible Root Causes
              </h3>
              <ul className="list-disc pl-5 text-sm text-gray-600">
                <li>
                  <span className="font-semibold text-gray-800">Switch Port Congestion</span>
                  <p className="text-xs mt-1 text-gray-500">
                    Correlation found with high CPU usage on "Switch-Lib-01".
                  </p>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: 4 Charts Grid (70% Width) */}
        <div className="col-span-8 grid grid-cols-2 gap-4">
           {/* Chart Placeholder 1 */}
           <div className="bg-white h-60 rounded-xl shadow-sm border border-gray-200 flex flex-col p-4">
              <h4 className="font-bold text-sm text-gray-600 mb-2">Lan Latency (ms)</h4>
              <div className="flex-1 bg-gray-50 border border-dashed rounded flex items-center justify-center text-gray-400 text-sm">
                [ Line Chart Placeholder ]
              </div>
           </div>

           {/* Chart Placeholder 2 */}
           <div className="bg-white h-60 rounded-xl shadow-sm border border-gray-200 flex flex-col p-4">
              <h4 className="font-bold text-sm text-gray-600 mb-2">Wifi Latency (ms)</h4>
              <div className="flex-1 bg-gray-50 border border-dashed rounded flex items-center justify-center text-gray-400 text-sm">
                [ Line Chart Placeholder ]
              </div>
           </div>

           {/* Chart Placeholder 3 */}
           <div className="bg-white h-60 rounded-xl shadow-sm border border-gray-200 flex flex-col p-4">
              <h4 className="font-bold text-sm text-gray-600 mb-2">Lan Internet Speed</h4>
              <div className="flex-1 bg-gray-50 border border-dashed rounded flex items-center justify-center text-gray-400 text-sm">
                [ Line Chart Placeholder ]
              </div>
           </div>

           {/* Chart Placeholder 4 */}
           <div className="bg-white h-60 rounded-xl shadow-sm border border-gray-200 flex flex-col p-4">
              <h4 className="font-bold text-sm text-gray-600 mb-2">Wifi Internet Speed</h4>
              <div className="flex-1 bg-gray-50 border border-dashed rounded flex items-center justify-center text-gray-400 text-sm">
                [ Line Chart Placeholder ]
              </div>
           </div>
        </div>

      </div>
    </div>
  );
};

export default ProbeDetails;