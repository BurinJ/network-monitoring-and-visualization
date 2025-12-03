import { useState } from 'react'; // Import useState
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Overview from './pages/Overview.jsx';
import Reports from './pages/Reports.jsx';
import ProbeDetails from './pages/ProbeDetails.jsx';
import Sidebar from './components/Layout/Sidebar.jsx';

// const Overview = () => <div className="p-8"><h2 className="text-2xl font-bold">Overview Dashboard</h2></div>;
// const Reports = () => <div className="p-8"><h2 className="text-2xl font-bold">Downed Probes Status</h2></div>;
// const ProbeDetails = () => <div className="p-8"><h2 className="text-2xl font-bold">Probe Details</h2></div>;

function App() {
  // 1. Create state to track sidebar status
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <Router>
      <div className="flex min-h-screen bg-gray-50">
        
        {/* 2. Pass the state and the toggle function to Sidebar */}
        <Sidebar isOpen={isSidebarOpen} toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />

        {/* 3. Adjust margin dynamically: ml-64 (open) vs ml-20 (closed) */}
        <div 
          className={`flex-1 transition-all duration-300 ease-in-out ${
            isSidebarOpen ? 'ml-64' : 'ml-20'
          }`}
        >
          {/* Header
          <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 sticky top-0 z-10">
            <h1 className="text-gray-700 font-semibold">Dashboard</h1>
          </header>
          */}

          <main>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/report" element={<Reports />} />
              <Route path="/probe/:id" element={<ProbeDetails />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;