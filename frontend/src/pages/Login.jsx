import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Lock, User } from 'lucide-react';

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simple Mock Authentication Logic
    // In a real app, this would call your backend API
    if (username === 'admin' && password === 'admin') {
      onLogin({ username: 'admin', role: 'admin', department: 'IT Dept' });
      navigate('/admin');
    } else if (username === 'staff' && password === 'staff') {
      onLogin({ username: 'staff', role: 'viewer', department: 'Faculty' });
      navigate('/');
    } else {
      setError('Invalid credentials');
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-teal-50">
      <div className="bg-white p-8 rounded-2xl shadow-lg border border-teal-100 w-96">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-teal-900 mb-2">KU Net</h1>
          <p className="text-slate-500 text-sm">Network Monitoring System</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Username</label>
            <div className="relative">
              <User className="absolute left-3 top-3 text-slate-400" size={18} />
              <input 
                type="text" 
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 transition-colors"
                placeholder="Enter username"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Password</label>
            <div className="relative">
              <Lock className="absolute left-3 top-3 text-slate-400" size={18} />
              <input 
                type="password" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 transition-colors"
                placeholder="••••••••"
              />
            </div>
          </div>

          {error && (
            <div className="text-red-500 text-sm text-center bg-red-50 p-2 rounded">
              {error}
            </div>
          )}

          <button 
            type="submit"
            className="w-full bg-teal-600 text-white py-2 rounded-lg font-bold hover:bg-teal-700 transition-colors shadow-md hover:shadow-lg"
          >
            Sign In
          </button>
        </form>

        <div className="mt-6 text-center text-xs text-slate-400">
          <p>Admin Login: admin / admin</p>
          <p>Staff Login: staff / staff</p>
        </div>
      </div>
    </div>
  );
};

export default Login;