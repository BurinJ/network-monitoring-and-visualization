import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';

// Pages
import NetworkStatus from './pages/NetworkStatus'; // Public Status Page
import CommandCenter from './pages/CommandCenter'; // Admin Dashboard
import Inspector from './pages/Inspector'; // Probe Details
import Trends from './pages/Trends'; // AI Trends
import Login from './pages/Login'; // Login Page
import AlertHistory from './pages/AlertHistory'; // Alert
import Settings from './pages/Settings'; // Settings
import ForecastDetails from './pages/ForecastDetails'; // Individual Network Forecasting

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [user, setUser] = useState(null);
  // Add loading state to prevent premature redirect
  const [authLoading, setAuthLoading] = useState(true);

  useEffect(() => {
    const savedUser = localStorage.getItem('kunet_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }
    setAuthLoading(false); // Mark check as complete
  }, []);

  const handleLogin = (userData) => {
    setUser(userData);
    localStorage.setItem('kunet_user', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('kunet_user');
  };

  // Prevent router from running until we've checked for a saved session
  if (authLoading) return <div className="p-10 text-teal-600 font-medium">Loading App...</div>;

  return (
    <Router>
      <div className="flex min-h-screen bg-teal-50">
        {user && (
          <Sidebar 
            isOpen={isSidebarOpen} 
            toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)} 
            user={user}
            onLogout={handleLogout}
          />
        )}

        <div className={`flex-1 transition-all duration-300 ease-in-out ${user ? (isSidebarOpen ? 'ml-64' : 'ml-20') : ''}`}>
          <main>
            <Routes>
              <Route path="/login" element={<Login onLogin={handleLogin} />} />
              <Route path="/" element={<NetworkStatus user={user} />} />
              
              <Route path="/admin" element={user ? <CommandCenter /> : <Navigate to="/login" />} />
              <Route path="/inspector/:probeId" element={user ? <Inspector /> : <Navigate to="/login" />} />
              <Route path="/forecast/:probeId" element={user ? <ForecastDetails /> : <Navigate to="/login" />} />
              <Route path="/trends" element={user ? <Trends /> : <Navigate to="/login" />} />
              <Route path="/history" element={user ? <AlertHistory /> : <Navigate to="/login" />} />
              <Route path="/settings" element={user ? <Settings /> : <Navigate to="/login" />} />

              <Route path="*" element={<div className="p-10 text-teal-800">404: Page Not Found</div>} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;