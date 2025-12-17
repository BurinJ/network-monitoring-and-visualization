import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';

// Import Renamed Pages
import NetworkStatus from './pages/NetworkStatus';
import CommandCenter from './pages/CommandCenter';
import Inspector from './pages/Inspector';
import Trends from './pages/Trends';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <Router>
      <div className="flex min-h-screen bg-gray-50">
        <Sidebar isOpen={isSidebarOpen} toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} />

        <div className={`flex-1 transition-all duration-300 ease-in-out ${isSidebarOpen ? 'ml-64' : 'ml-20'}`}>
          
          <main>
            <Routes>
              {/* 1. Public Status Page (Default) */}
              <Route path="/" element={<NetworkStatus />} />
              
              {/* 2. Admin Dashboard */}
              <Route path="/admin" element={<CommandCenter />} />
              
              {/* 3. Deep Dive (Inspector) */}
              <Route path="/inspector/:probeId" element={<Inspector />} />
              
              {/* 4. Strategic View */}
              <Route path="/trends" element={<Trends />} />

              {/* Redirect old routes if necessary */}
              <Route path="/overview" element={<Navigate to="/admin" replace />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;