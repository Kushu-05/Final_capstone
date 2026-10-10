import React from 'react';
import { Utensils, Cpu, Search, TrendingUp, BarChart3, Award, CheckCircle } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, config }) {
  const isCloud = Boolean(
    config?.anthropic_configured ||
    config?.nvidia_configured ||
    config?.openai_configured ||
    config?.gemini_configured ||
    config?.openrouter_configured
  );

  const getEngineLabel = () => {
    if (config?.active_provider === 'anthropic') return 'Claude 3.5 Sonnet (Primary)';
    if (config?.active_provider === 'nvidia') return 'NVIDIA GPT (Secondary)';
    if (config?.active_provider === 'openai') return 'OpenAI GPT';
    if (config?.active_provider === 'gemini') return 'Google Gemini';
    if (config?.active_provider === 'openrouter') return 'OpenRouter';
    return isCloud ? 'Cloud AI Active' : 'Operational (Local AI)';
  };

  return (
    <header className="sticky top-0 z-50 bg-slate-900/90 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & App Name */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Utensils className="w-5 h-5 text-slate-950 font-bold" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-extrabold text-lg text-white tracking-tight">GastroOps</span>
                <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded-full">
                  RESTAURANT AI
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">AI-Powered Restaurant Operations & Review Assistant</p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            <button
              onClick={() => setActiveTab('multi-agent')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'multi-agent'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Cpu className="w-4 h-4 text-emerald-400" />
              <span>Operations Assistant</span>
            </button>

            <button
              onClick={() => setActiveTab('search')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'search'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Search className="w-4 h-4 text-sky-400" />
              <span>Review Explorer</span>
            </button>

            <button
              onClick={() => setActiveTab('trends')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'trends'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <TrendingUp className="w-4 h-4 text-amber-400" />
              <span>Complaint Trends</span>
            </button>

            <button
              onClick={() => setActiveTab('csat')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'csat'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <BarChart3 className="w-4 h-4 text-purple-400" />
              <span>Customer Satisfaction</span>
            </button>

            <button
              onClick={() => setActiveTab('evaluation')}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all ${
                activeTab === 'evaluation'
                  ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm'
                  : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
              }`}
            >
              <Award className="w-4 h-4 text-rose-400" />
              <span>Quality & Feedback</span>
            </button>
          </nav>

          {/* Clean Non-Technical System Status */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-2 px-3 py-1 bg-slate-800/60 border border-slate-700/60 rounded-full text-xs">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-slate-300 font-medium hidden md:inline">
                {getEngineLabel()}
              </span>
              <span className="text-emerald-400 text-[10px] font-bold">ONLINE</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
