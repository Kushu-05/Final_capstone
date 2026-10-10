import React, { useState, useEffect } from 'react';
import { Search, Filter, Star, MapPin, Sparkles, CheckCircle, FileText } from 'lucide-react';
import { runSearch } from '../api';

export default function SearchExplorerView({ metadata }) {
  const [query, setQuery] = useState('slow service and cold food');
  const [restaurant, setRestaurant] = useState('all');
  const [category, setCategory] = useState('all');
  const [city, setCity] = useState('all');
  const [minStars, setMinStars] = useState(1);
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [keyReasons, setKeyReasons] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await runSearch({
        query,
        alpha: 0.5,
        top_k: 10,
        business_id: restaurant === 'all' ? null : restaurant,
        category: category === 'all' ? null : category,
        city: city === 'all' ? null : city,
        min_stars: parseInt(minStars, 10)
      });
      setResults(data.results || []);
      setSummary(data.executive_summary || null);
      setKeyReasons(data.key_reasons || []);
    } catch (err) {
      setError(err.message || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleSearch();
  }, []);

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-1">
          <Search className="w-5 h-5 text-sky-400" />
          Customer Review & Complaint Explorer
        </h2>
        <p className="text-xs text-slate-400 mb-5">
          Quickly search and inspect customer experiences, recurring complaints, and ratings across restaurant locations.
        </p>

        {/* Input Bar */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3.5" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search by issue or dish (e.g., 'lukewarm soup', 'wait time', 'rude manager')..."
              className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-3 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-sky-500"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-3 bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold rounded-xl text-xs transition flex items-center gap-2 disabled:opacity-50 shadow-md shadow-sky-500/20"
          >
            {loading ? 'Searching...' : 'Search Reviews'}
          </button>
        </div>

        {/* Filter Controls */}
        <div className="mt-5 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          {/* Restaurant Filter */}
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
            <label className="text-slate-400 font-medium block mb-1">Restaurant</label>
            <select
              value={restaurant}
              onChange={(e) => setRestaurant(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white"
            >
              <option value="all">All Restaurants</option>
              {metadata?.restaurants?.map((r, i) => (
                <option key={i} value={r}>{r}</option>
              ))}
            </select>
          </div>

          {/* Category Filter */}
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
            <label className="text-slate-400 font-medium block mb-1">Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white"
            >
              <option value="all">All Categories</option>
              {metadata?.categories?.map((c, i) => (
                <option key={i} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* City Filter */}
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800">
            <label className="text-slate-400 font-medium block mb-1">City</label>
            <select
              value={city}
              onChange={(e) => setCity(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white"
            >
              <option value="all">All Cities</option>
              {metadata?.cities?.map((c, i) => (
                <option key={i} value={c}>{c}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Executive Operational Summary & Direct Answer */}
      {summary && (
        <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-sky-500/30 rounded-2xl p-6 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-white text-sm">
                  Executive Review Summary & Direct Answer
                </h3>
                <p className="text-xs text-slate-400">
                  Synthesized from customer reviews for query: <span className="text-sky-300 italic">"{query}"</span>
                </p>
              </div>
            </div>
            <span className="text-[11px] font-semibold text-sky-400 bg-sky-500/10 border border-sky-500/20 px-3 py-1 rounded-full flex items-center gap-1.5 self-start sm:self-auto">
              <FileText className="w-3.5 h-3.5" />
              {results.length} Review Evidence Cases
            </span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4">
            <div className="text-[11px] uppercase font-bold text-sky-400 mb-1 flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5" />
              Synthesized Operational Answer
            </div>
            <p className="text-xs text-slate-200 leading-relaxed">
              {summary}
            </p>
          </div>

          {keyReasons && keyReasons.length > 0 && (
            <div className="space-y-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Key Operational Reasons Identified:
              </span>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                {keyReasons.map((reason, idx) => (
                  <div key={idx} className="flex items-start gap-2 p-2.5 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-300">
                    <span className="w-4 h-4 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      {idx + 1}
                    </span>
                    <span className="leading-snug">{reason}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Results Header */}
      <div className="flex items-center justify-between text-xs text-slate-400 px-1">
        <span>Found <strong className="text-white">{results.length}</strong> customer reviews as evidence</span>
        <span>Search Engine: <strong className="text-emerald-400">Active</strong></span>
      </div>

      {/* Results List */}
      {results.length === 0 && !loading ? (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center space-y-2.5">
          <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 mb-1">
            <CheckCircle className="w-6 h-6" />
          </div>
          <h3 className="font-bold text-sm text-white">
            {restaurant !== 'all' ? `No Complaints Found for ${restaurant}` : 'No Matching Reviews Found'}
          </h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            {restaurant !== 'all'
              ? `Customer reviews for ${restaurant} do not contain reports matching "${query}".`
              : `No customer reviews found matching "${query}".`}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {results.map((r, idx) => (
            <div key={idx} className="bg-slate-900 border border-slate-800 rounded-2xl p-5 hover:border-slate-700 transition space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded bg-sky-500/15 border border-sky-500/30 text-[10px] font-bold text-sky-400">
                      Evidence Case #{idx + 1}
                    </span>
                    <h4 className="font-bold text-sm text-white">{r.restaurant_name}</h4>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs text-slate-500 mt-1">
                    <MapPin className="w-3 h-3 text-slate-400" />
                    <span>{[r.city, r.state].filter(Boolean).join(', ')}</span>
                    {r.date && (
                      <>
                        <span>•</span>
                        <span>{r.date?.slice(0, 10)}</span>
                      </>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-1 bg-amber-500/10 border border-amber-500/30 px-2.5 py-1 rounded-full text-amber-400 font-black text-xs">
                  <Star className="w-3.5 h-3.5 fill-amber-400 text-amber-400" />
                  <span>{r.stars}.0</span>
                </div>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed italic">
                "{r.text}"
              </p>

              <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
                <span className="px-2 py-0.5 rounded-md bg-slate-950 border border-slate-800 text-slate-400 font-medium">
                  {r.category || 'Customer Review'}
                </span>
                <span className="text-sky-400 font-bold bg-sky-500/10 px-2 py-0.5 rounded-full border border-sky-500/20 text-[10px]">
                  {Math.round((r.hybrid_score || 0.85) * 100)}% Match
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
