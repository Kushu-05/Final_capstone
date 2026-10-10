import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, AlertTriangle, Calendar, BarChart3, ArrowUpRight, 
  ArrowDownRight, Minus, RefreshCw, Layers 
} from 'lucide-react';
import { 
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  BarChart, Bar, CartesianGrid, Legend 
} from 'recharts';
import { fetchTrends } from '../api';

export default function TrendPredictionView({ metadata }) {
  const [selectedRestaurant, setSelectedRestaurant] = useState('all');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const loadTrends = async (bizId) => {
    setLoading(true);
    try {
      const res = await fetchTrends({
        business_id: bizId === 'all' ? null : bizId
      });
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTrends(selectedRestaurant);
  }, [selectedRestaurant]);

  // Format category distribution for BarChart
  const categoryChartData = data?.category_distribution
    ? Object.entries(data.category_distribution).map(([cat, count]) => ({
        category: cat,
        count
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-amber-400" />
            Complaint Trend Prediction & Forecasting
          </h2>
          <p className="text-xs text-slate-400">
            Temporal time-series regression and category risk forecasting derived from historical Yelp timestamps.
          </p>
        </div>

        {/* Restaurant Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">Restaurant:</span>
          <select
            value={selectedRestaurant}
            onChange={(e) => setSelectedRestaurant(e.target.value)}
            className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-white focus:outline-none focus:border-amber-500"
          >
            <option value="all">All Restaurants Aggregated</option>
            {metadata?.restaurants?.map((r, i) => (
              <option key={i} value={r}>{r}</option>
            ))}
          </select>
        </div>
      </div>

      {loading ? (
        <div className="p-12 text-center text-slate-500 text-xs flex items-center justify-center gap-2">
          <RefreshCw className="w-4 h-4 animate-spin text-amber-400" />
          <span>Calculating temporal regression trajectories...</span>
        </div>
      ) : (
        <>
          {/* Key Metrics KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Trajectory */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Overall Trajectory</span>
              <div className="flex items-center gap-2">
                {data?.trend_slope > 0 ? (
                  <ArrowUpRight className="w-5 h-5 text-rose-400" />
                ) : data?.trend_slope < 0 ? (
                  <ArrowDownRight className="w-5 h-5 text-emerald-400" />
                ) : (
                  <Minus className="w-5 h-5 text-sky-400" />
                )}
                <span className={`text-base font-black ${
                  data?.trajectory_color === 'red' ? 'text-rose-400' :
                  data?.trajectory_color === 'orange' ? 'text-amber-400' :
                  data?.trajectory_color === 'green' ? 'text-emerald-400' : 'text-sky-400'
                }`}>
                  {data?.overall_trajectory}
                </span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Slope: <strong className="text-white font-mono">{data?.trend_slope > 0 ? `+${data?.trend_slope}` : data?.trend_slope}</strong> / month
              </div>
            </div>

            {/* Next Month Forecast */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Next Month Forecast</span>
              <div className="text-2xl font-black text-white">
                ~{data?.projected_next_month_complaints} <span className="text-xs font-normal text-slate-400">cases</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Projected complaint volume at current rate
              </div>
            </div>

            {/* Top Risk Domain */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Top Surge Domain</span>
              <div className="text-base font-bold text-amber-400 truncate">
                {data?.top_surge_category}
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Highest complaint recurrence in current window
              </div>
            </div>

            {/* Time Span */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Data Timeline</span>
              <div className="text-sm font-bold text-white flex items-center gap-1.5">
                <Calendar className="w-4 h-4 text-emerald-400" />
                <span>2024 - 2026 Historical</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                {data?.historical_monthly_data?.length || 0} discrete monthly buckets
              </div>
            </div>
          </div>

          {/* Time Series Area Chart */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <h3 className="font-bold text-sm text-white uppercase tracking-wider flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-emerald-400" />
              Monthly Complaint Volume & Total Reviews Trend
            </h3>
            
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data?.historical_monthly_data || []}>
                  <defs>
                    <linearGradient id="complaintGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="totalGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="month" stroke="#64748b" textAnchor="end" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                    itemStyle={{ color: '#f8fafc' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Area type="monotone" dataKey="complaints" name="Complaints" stroke="#f43f5e" fillOpacity={1} fill="url(#complaintGrad)" />
                  <Area type="monotone" dataKey="total_reviews" name="Total Reviews" stroke="#10b981" fillOpacity={1} fill="url(#totalGrad)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Category Distribution Bar Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="font-bold text-sm text-white uppercase tracking-wider flex items-center gap-2">
                <Layers className="w-4 h-4 text-purple-400" />
                Complaints by Operational Domain
              </h3>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categoryChartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="category" stroke="#64748b" tick={{ fontSize: 9 }} angle={-15} textAnchor="end" />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                      itemStyle={{ color: '#f8fafc' }}
                    />
                    <Bar dataKey="count" name="Reported Incidents" fill="#a855f7" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Strategic Insights */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="font-bold text-sm text-white uppercase tracking-wider flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Predictive Operations Insights
              </h3>
              <div className="space-y-3">
                {data?.insights?.map((ins, i) => (
                  <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300">
                    <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      {i + 1}
                    </span>
                    <span className="leading-relaxed">{ins}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
