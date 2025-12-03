import React from 'react';

const Overview = () => {
  return (
    <div className="p-6 bg-gray-50 min-h-screen text-gray-800">
      <h1 className="text-2xl font-bold mb-6">Overview</h1>
      
      {/* Top Row Grid */}
      <div className="grid grid-cols-12 gap-6 mb-6">
        
        {/* Left: Donut Chart Placeholder */}
        <div className="col-span-3 bg-white p-4 rounded-xl shadow-sm border border-gray-200 h-80 flex flex-col items-center justify-center">
          <h3 className="font-semibold text-gray-500 mb-2">Probe Status</h3>
          <div className="w-32 h-32 rounded-full border-4 border-dashed border-gray-300 flex items-center justify-center text-xs text-gray-400">
            [Donut Chart]
          </div>
          <div className="mt-4 text-sm text-center">
            <span className="text-green-500 font-bold">41 Normal</span> • <span className="text-red-500 font-bold">55 Downed</span>
          </div>
        </div>

        {/* Middle: Bar Chart & Map Placeholders */}
        <div className="col-span-5 flex flex-col gap-6">
           <div className="h-40 bg-white p-4 rounded-xl shadow-sm border border-gray-200 flex items-center justify-center">
             <span className="text-gray-400 font-medium">[Department Status Bar Chart]</span>
           </div>
           <div className="h-34 bg-gray-200 rounded-xl border border-gray-300 flex items-center justify-center grow">
             <span className="text-gray-500 font-medium">[Network Map Visualization]</span>
           </div>
        </div>

        {/* Right: Performance Table Placeholder */}
        <div className="col-span-4 bg-white rounded-xl shadow-sm border border-gray-200 p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold">WLAN Speed</h3>
            <span className="text-xs text-blue-500 cursor-pointer">View All</span>
          </div>
          {/* Static Table Placeholder */}
          <div className="border rounded-lg overflow-hidden">
            <div className="grid grid-cols-3 bg-gray-100 p-2 text-sm font-semibold text-gray-600">
              <div>Probe</div><div>Down</div><div>Up</div>
            </div>
            {[1, 2, 3, 4, 5].map((item) => (
              <div key={item} className="grid grid-cols-3 p-2 text-sm border-t border-gray-100">
                <div>Probe {item}</div>
                <div>1{item}2 Mbps</div>
                <div>8{item} Mbps</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Row: Downtime Table */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200">
        <h3 className="font-bold mb-4">Downtime Report</h3>
        <div className="w-full h-32 bg-gray-50 border-2 border-dashed border-gray-200 rounded-lg flex items-center justify-center text-gray-400">
          [Downtime Table Data Placeholder]
        </div>
      </div>
    </div>
  );
};

export default Overview;