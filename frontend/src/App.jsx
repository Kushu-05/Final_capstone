import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MultiAgentView from './components/MultiAgentView';
import SearchExplorerView from './components/SearchExplorerView';
import TrendPredictionView from './components/TrendPredictionView';
import CSATPredictorView from './components/CSATPredictorView';
import DeepEvalFeedbackView from './components/DeepEvalFeedbackView';
import { fetchConfig, fetchMetadata, fetchHealth } from './api';

export default function App() {
  const [activeTab, setActiveTab] = useState('multi-agent');
  const [config, setConfig] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [health, setHealth] = useState(null);

  const loadData = async () => {
    try {
      const cfg = await fetchConfig().catch(err => {
        console.warn('fetchConfig warning:', err);
        return null;
      });
      const meta = await fetchMetadata().catch(err => {
        console.warn('fetchMetadata warning:', err);
        return null;
      });
      const hlth = await fetchHealth().catch(err => {
        console.warn('fetchHealth warning:', err);
        return null;
      });

      if (cfg) setConfig(cfg);
      if (meta) setMetadata(meta);
      if (hlth) setHealth(hlth);
    } catch (err) {
      console.error('Failed to load initial app data:', err);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        config={config}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'multi-agent' && <MultiAgentView metadata={metadata} />}
        {activeTab === 'search' && <SearchExplorerView metadata={metadata} />}
        {activeTab === 'trends' && <TrendPredictionView metadata={metadata} />}
        {activeTab === 'csat' && <CSATPredictorView />}
        {activeTab === 'evaluation' && <DeepEvalFeedbackView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>AI-Powered Restaurant Operations & Review Assistant</span>
          <span className="flex items-center gap-1.5 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            System Operational • Yelp Open Dataset
          </span>
        </div>
      </footer>
    </div>
  );
}
