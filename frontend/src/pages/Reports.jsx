import React from 'react';

const Reports = () => {
  return (
    <div className="p-6 bg-gray-50 min-h-screen text-gray-800">
      
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Downed Probe Status Report</h1>
        <p className="text-gray-500 text-sm mt-1">
          This downtime report is flagged by Anomaly Detection (AI Engine)
        </p>
      </div>

      <div className="grid grid-cols-12 gap-8">
        
        {/* Left Column: List of Probes */}
        <div className="col-span-5 bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="grid grid-cols-2 bg-gray-100 p-4 font-bold text-gray-700 border-b">
            <div>Probe Name</div>
            <div>Downtime Duration</div>
          </div>
          {/* List items */}
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div key={i} className="grid grid-cols-2 p-4 border-b last:border-0 hover:bg-gray-50 transition">
              <div className="font-medium">Probe {i}</div>
              <div className="text-gray-500 text-sm">02 days 04 hours 12 m</div>
            </div>
          ))}
        </div>

        {/* Right Column: Map Visualization */}
        <div className="col-span-7">
          <h3 className="text-lg font-semibold mb-4 text-center text-gray-600">
            Where the downed probes are located
          </h3>
          <div className="w-full h-[500px] bg-slate-200 rounded-xl border-2 border-gray-300 flex items-center justify-center relative shadow-inner">
             <span className="text-gray-500 font-bold text-lg">[ Interactive Map Placeholder ]</span>
             {/* Fake "Pins" to make it look real */}
             <div className="absolute top-1/4 left-1/4 w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
             <div className="absolute top-1/2 left-1/2 w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
             <div className="absolute bottom-1/3 right-1/4 w-4 h-4 bg-red-500 rounded-full border-2 border-white shadow-lg"></div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Reports;