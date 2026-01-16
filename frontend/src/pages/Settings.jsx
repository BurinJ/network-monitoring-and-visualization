import React, { useState, useEffect } from 'react';
import { Save, RefreshCw, Cpu, ChevronDown } from 'lucide-react';
import { API_BASE_URL } from '../config'; 

const Settings = () => {
  const [probes, setProbes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [training, setTraining] = useState(false);
  const [trainMsg, setTrainMsg] = useState('');
  
  // Selection State
  const [selectedProbeId, setSelectedProbeId] = useState('');
  const [editName, setEditName] = useState('');

  useEffect(() => {
    fetchProbes();
  }, []);

  const fetchProbes = () => {
    setLoading(true);
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/settings/probes`)
      .then(res => res.json())
      .then(data => {
        setProbes(data);
        if (data.length > 0 && !selectedProbeId) {
            setSelectedProbeId(data[0].id);
            setEditName(data[0].name);
        }
        setLoading(false);
      })
      .catch(err => setLoading(false));
  };

  const handleSelectProbe = (e) => {
    const id = e.target.value;
    setSelectedProbeId(id);
    const probe = probes.find(p => p.id === id);
    if (probe) setEditName(probe.name);
  };

  const handleSave = () => {
    if (!selectedProbeId) return;
    setSaving(true);
    
    // Optimistic Update
    setProbes(prev => prev.map(p => p.id === selectedProbeId ? { ...p, name: editName } : p));

    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/settings/probe`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: selectedProbeId, name: editName })
    })
    .then(() => setSaving(false))
    .catch(() => setSaving(false));
  };

  const handleRetrain = () => {
    setTraining(true);
    setTrainMsg('Training in progress... This may take a few minutes.');
    const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
    fetch(`${baseUrl}/admin/train`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        setTraining(false);
        setTrainMsg(data.message || 'Training Complete!');
      })
      .catch(err => {
        setTraining(false);
        setTrainMsg('Error during training.');
      });
  };

  return (
    <div className="p-8 bg-teal-50 min-h-screen">
      <h1 className="text-3xl font-bold text-teal-900 mb-8">Admin Settings</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Probe Naming Section */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-teal-800">Probe Configuration</h2>
            <button onClick={fetchProbes} className="p-2 text-teal-500 hover:bg-teal-50 rounded-full transition">
              <RefreshCw size={18} />
            </button>
          </div>
          
          {loading ? (
            <div className="text-center text-gray-400 py-8">Loading settings...</div>
          ) : (
            <div className="space-y-6">
                <div>
                    <label className="block text-sm font-semibold text-teal-700 mb-2">Select Probe to Edit</label>
                    <div className="relative">
                        <select 
                            value={selectedProbeId}
                            onChange={handleSelectProbe}
                            className="w-full p-3 border border-gray-200 rounded-lg bg-gray-50 text-slate-700 font-medium appearance-none focus:outline-none focus:ring-2 focus:ring-teal-500"
                        >
                            {probes.map(p => (
                                <option key={p.id} value={p.id}>{p.id} ({p.name})</option>
                            ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" size={18} />
                    </div>
                </div>

                <div>
                    <label className="block text-sm font-semibold text-teal-700 mb-2">Friendly Name</label>
                    <input 
                      type="text" 
                      value={editName} 
                      onChange={(e) => setEditName(e.target.value)}
                      className="w-full bg-white border border-gray-200 rounded-lg px-3 py-3 text-slate-800 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500"
                      placeholder="Enter new display name..."
                    />
                    <p className="text-xs text-gray-400 mt-2">
                        This name will be displayed across all dashboards instead of the raw ID.
                    </p>
                </div>

                <div className="pt-4 border-t border-gray-100 flex justify-end">
                    <button 
                        onClick={handleSave}
                        disabled={saving || !selectedProbeId}
                        className="flex items-center gap-2 px-6 py-2.5 bg-teal-600 text-white font-bold rounded-lg hover:bg-teal-700 transition disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
                    >
                        <Save size={18} />
                        {saving ? 'Saving...' : 'Save Changes'}
                    </button>
                </div>
            </div>
          )}
        </div>

        {/* AI Management Section */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-teal-100 h-fit">
          <h2 className="text-xl font-bold text-teal-800 mb-4 flex items-center gap-2">
            <Cpu size={24} /> AI Model Management
          </h2>
          <p className="text-sm text-slate-500 mb-6">
            Retrain the Anomaly Detection and Forecast models using the latest historical data from Prometheus. 
            Recommended to run once a week or after adding new probes.
          </p>
          
          <button 
            onClick={handleRetrain}
            disabled={training}
            className={`w-full py-3 rounded-lg font-bold text-white transition-all flex items-center justify-center gap-2
              ${training ? 'bg-gray-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700 shadow-md'}
            `}
          >
            {training ? <RefreshCw size={20} className="animate-spin" /> : <Cpu size={20} />}
            {training ? 'Training Models...' : 'Retrain AI Models Now'}
          </button>
          
          {trainMsg && (
            <div className={`mt-4 p-3 rounded text-sm text-center ${trainMsg.includes('Error') ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-700'}`}>
              {trainMsg}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Settings;