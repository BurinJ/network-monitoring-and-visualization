import React, { useState } from 'react';
import { Settings as SettingsIcon, RefreshCw, CheckCircle, AlertTriangle, Database } from 'lucide-react';
import { API_BASE_URL } from '../config';

const Settings = () => {
  const [trainingStatus, setTrainingStatus] = useState('idle'); // idle, training, success, error
  const [message, setMessage] = useState('');

  const handleRetrain = async () => {
    setTrainingStatus('training');
    setMessage('Starting training process... this may take a minute.');
    
    try {
        const baseUrl = typeof API_BASE_URL !== 'undefined' ? API_BASE_URL : 'http://localhost:5000/api';
        // Note: You need to add this endpoint to server/app.py
        const res = await fetch(`${baseUrl}/admin/train`, { method: 'POST' });
        const data = await res.json();
        
        if (res.ok) {
            setTrainingStatus('success');
            setMessage(`Success: ${data.message}`);
        } else {
            setTrainingStatus('error');
            setMessage(`Error: ${data.error || 'Failed to train models'}`);
        }
    } catch (err) {
        setTrainingStatus('error');
        setMessage(`Connection Error: ${err.message}. Is the backend running?`);
    }
  };

  return (
    <div className="p-6 bg-teal-50 min-h-screen">
      <h1 className="text-2xl font-bold text-teal-900 mb-6 flex items-center gap-3">
        <SettingsIcon className="text-teal-600" /> System Settings
      </h1>

      <div className="bg-white rounded-xl shadow-sm border border-teal-100 overflow-hidden max-w-3xl">
        {/* Header */}
        <div className="p-6 border-b border-teal-50 bg-teal-50/50">
            <h2 className="text-lg font-semibold text-teal-800 flex items-center gap-2">
                <Database size={20} /> AI Model Management
            </h2>
            <p className="text-sm text-slate-500 mt-1">
                Manage the machine learning models used for Anomaly Detection (Isolation Forest) and Trend Forecasting (LSTM/Random Forest).
            </p>
        </div>

        {/* Content */}
        <div className="p-6">
            <div className="flex items-start gap-6">
                <div className="flex-1">
                    <h3 className="font-medium text-slate-700 mb-2">Retrain Models</h3>
                    <p className="text-sm text-slate-500 mb-4 leading-relaxed">
                        This action will fetch the latest 14 days of data from Prometheus, recalculate baselines (Max Speed) for all probes, 
                        and retrain the AI models to adapt to recent network patterns.
                        <br/><br/>
                        <span className="text-orange-600 bg-orange-50 px-2 py-1 rounded text-xs font-bold border border-orange-100">
                            Recommended: Run weekly or after adding new probes.
                        </span>
                    </p>
                </div>

                <div className="flex flex-col items-end gap-3 min-w-[180px]">
                    <button
                        onClick={handleRetrain}
                        disabled={trainingStatus === 'training'}
                        className={`flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-bold text-white transition-all shadow-sm w-full
                            ${trainingStatus === 'training' 
                                ? 'bg-slate-400 cursor-not-allowed' 
                                : 'bg-teal-600 hover:bg-teal-700 hover:shadow-md active:transform active:scale-95'}
                        `}
                    >
                        <RefreshCw size={20} className={trainingStatus === 'training' ? 'animate-spin' : ''} />
                        {trainingStatus === 'training' ? 'Training...' : 'Retrain Now'}
                    </button>
                </div>
            </div>

            {/* Status Message Area */}
            {trainingStatus !== 'idle' && (
                <div className={`mt-6 p-4 rounded-lg border flex items-start gap-3 animate-in fade-in slide-in-from-top-2 duration-300 ${
                    trainingStatus === 'success' ? 'bg-emerald-50 border-emerald-200 text-emerald-800' :
                    trainingStatus === 'error' ? 'bg-red-50 border-red-200 text-red-800' :
                    'bg-blue-50 border-blue-200 text-blue-800'
                }`}>
                    <div className="mt-0.5">
                        {trainingStatus === 'success' && <CheckCircle size={20} />}
                        {trainingStatus === 'error' && <AlertTriangle size={20} />}
                        {trainingStatus === 'training' && <RefreshCw size={20} className="animate-spin" />}
                    </div>
                    <div>
                        <span className="font-bold block text-sm uppercase mb-1">
                            {trainingStatus === 'success' ? 'Training Complete' : 
                             trainingStatus === 'error' ? 'Training Failed' : 'Processing'}
                        </span>
                        <span className="text-sm opacity-90">{message}</span>
                    </div>
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default Settings;