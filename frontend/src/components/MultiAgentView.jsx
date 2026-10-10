import React, { useState } from 'react';
import { 
  Search, Sliders, ShieldCheck, AlertTriangle, CheckCircle, Clock, 
  Sparkles, ThumbsUp, ThumbsDown, MessageSquare, ChevronRight, Activity, 
  FileText, Award, Compass, Sparkle, Star, AlertCircle
} from 'lucide-react';
import { runMultiAgent, submitFeedback } from '../api';

const SAMPLE_QUERIES = [
  "Why are customers giving low ratings, and what are the primary reasons?",
  "Customers are repeatedly complaining about slow service. What are the specific reasons and evidence?",
  "Food quality complaints: What issues are customers reporting regarding cold food or bad taste?",
  "Did anyone report getting sick or experiencing food poisoning after dining here?"
];

export default function MultiAgentView({ metadata }) {
  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [selectedRestaurant, setSelectedRestaurant] = useState('all');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedCity, setSelectedCity] = useState('all');
  const [minStars, setMinStars] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Feedback State
  const [feedbackHelpful, setFeedbackHelpful] = useState(true);
  const [feedbackRating, setFeedbackRating] = useState(5);
  const [feedbackNotes, setFeedbackNotes] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleRun = async (qToRun) => {
    const activeQuery = qToRun || query;
    if (!activeQuery.trim()) return;

    setLoading(true);
    setError(null);
    setFeedbackSubmitted(false);

    try {
      const payload = {
        query: activeQuery,
        alpha: 0.5,
        top_k: 5,
        business_id: selectedRestaurant === 'all' ? null : selectedRestaurant,
        category: selectedCategory === 'all' ? null : selectedCategory,
        city: selectedCity === 'all' ? null : selectedCity,
        min_stars: parseInt(minStars, 10)
      };

      const data = await runMultiAgent(payload);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Execution error');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedbackSubmit = async (e) => {
    e.preventDefault();
    if (!result) return;
    try {
      await submitFeedback({
        query: result.query,
        category: result.classification?.primary_category || 'General',
        is_helpful: feedbackHelpful,
        rating: feedbackRating,
        manager_notes: feedbackNotes,
        accepted_actions: result.recommendations?.immediate_actions || []
      });
      setFeedbackSubmitted(true);
    } catch (err) {
      alert('Error recording feedback: ' + err.message);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Hero / Query Input */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <Compass className="w-5 h-5 text-emerald-400" />
              Restaurant Operations Assistant
            </h2>
            <p className="text-xs text-slate-400">
              Enter any customer issue or operational question to retrieve relevant reviews, diagnose root causes, and get corrective action plans.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-400 bg-slate-950/80 px-3 py-1.5 rounded-xl border border-slate-800">
            <span className="font-semibold text-emerald-400">{metadata?.total_reviews ? metadata.total_reviews.toLocaleString() : '1,000'}</span> Historical Reviews
            <span>•</span>
            <span className="font-semibold text-sky-400">{metadata?.total_restaurants ? metadata.total_restaurants.toLocaleString() : '750'}</span> Restaurants Indexed
          </div>
        </div>

        {/* Input Bar */}
        <div className="relative">
          <textarea
            rows={3}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="E.g., Customers are repeatedly complaining about slow service and cold food. What similar complaints have been reported before?"
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 shadow-inner resize-none"
          />
          <button
            onClick={() => handleRun()}
            disabled={loading}
            className="absolute bottom-3 right-3 px-5 py-2 bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 font-bold rounded-lg text-xs transition flex items-center gap-2 shadow-lg shadow-emerald-500/20 disabled:opacity-50"
          >
            {loading ? (
              <span className="flex items-center gap-1.5">
                <span className="w-3.5 h-3.5 border-2 border-slate-950 border-t-transparent rounded-full animate-spin"></span>
                Analyzing Issue...
              </span>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Analyze & Solve</span>
              </>
            )}
          </button>
        </div>

        {/* Sample Prompt Chips */}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-[11px] font-semibold text-slate-500 uppercase">Common Questions:</span>
          {SAMPLE_QUERIES.map((q, idx) => (
            <button
              key={idx}
              onClick={() => {
                setQuery(q);
                handleRun(q);
              }}
              className="text-[11px] text-slate-300 hover:text-white bg-slate-800/60 hover:bg-slate-800 px-2.5 py-1 rounded-lg border border-slate-700/50 transition truncate max-w-xs"
              title={q}
            >
              {q.length > 45 ? q.slice(0, 45) + '...' : q}
            </button>
          ))}
        </div>

        {/* Metadata Filters */}
        <div className="mt-5 pt-4 border-t border-slate-800/80 grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          {/* Restaurant Filter */}
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex flex-col justify-between">
            <label className="text-slate-400 font-medium mb-1">Restaurant Filter</label>
            <select
              value={selectedRestaurant}
              onChange={(e) => setSelectedRestaurant(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="all">All Restaurants ({metadata?.restaurants?.length || 0})</option>
              {metadata?.restaurants?.map((r, i) => (
                <option key={i} value={r}>{r}</option>
              ))}
            </select>
          </div>

          {/* Category Filter */}
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex flex-col justify-between">
            <label className="text-slate-400 font-medium mb-1">Category Filter</label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="all">All Categories</option>
              {metadata?.categories?.map((c, i) => (
                <option key={i} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* City & Rating Filter */}
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex flex-col justify-between">
            <label className="text-slate-400 font-medium mb-1">Location & Min Stars</label>
            <div className="flex gap-2">
              <select
                value={selectedCity}
                onChange={(e) => setSelectedCity(e.target.value)}
                className="w-1/2 bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
              >
                <option value="all">All Cities</option>
                {metadata?.cities?.map((city, i) => (
                  <option key={i} value={city}>{city}</option>
                ))}
              </select>
              <select
                value={minStars}
                onChange={(e) => setMinStars(e.target.value)}
                className="w-1/2 bg-slate-900 border border-slate-800 rounded-lg p-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
              >
                <option value="1">★ 1+ Stars</option>
                <option value="2">★ 2+ Stars</option>
                <option value="3">★ 3+ Stars</option>
                <option value="4">★ 4+ Stars</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-sm flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 flex-shrink-0" />
          <div>
            <div className="font-bold">
              {typeof error === 'string' && (error.toLowerCase().includes('connect') || error.toLowerCase().includes('fetch') || error.toLowerCase().includes('network') || error.toLowerCase().includes('server'))
                ? 'System Connection Notice'
                : 'Validation Notice'}
            </div>
            <div className="text-xs text-rose-200/80">{error}</div>
          </div>
        </div>
      )}

      {/* Results Container */}
      {result && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* 1. Executive Review Summary & Direct Answer Card */}
          <div className="bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-emerald-500/40 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl -mr-20 -mt-20 pointer-events-none" />
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-emerald-500/20 to-teal-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-400 shadow-inner">
                  <Sparkles className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-white text-base flex items-center gap-2">
                    Executive Review Summary & Direct Answer
                  </h3>
                  <p className="text-xs text-slate-400">
                    Direct answer synthesized from customer reviews to: <span className="text-emerald-300 italic font-medium">"{result.query}"</span>
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 rounded-full flex items-center gap-1.5">
                  <FileText className="w-3.5 h-3.5" />
                  {result.retrieved_reviews?.length || 0} Evidence Reviews
                </span>
                {result.target_restaurant && (
                  <span className="text-[11px] font-semibold text-sky-400 bg-sky-500/10 border border-sky-500/20 px-3 py-1 rounded-full">
                    {result.target_restaurant}
                  </span>
                )}
                {result.classification?.primary_category && (
                  <span className="text-[11px] font-semibold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-3 py-1 rounded-full">
                    {result.classification.primary_category}
                  </span>
                )}
              </div>
            </div>

            {/* Synthesized Answer Paragraph */}
            <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4.5 mb-4 shadow-inner">
              <div className="text-[11px] uppercase font-bold tracking-wider text-emerald-400 mb-1.5 flex items-center gap-1.5">
                <CheckCircle className="w-3.5 h-3.5" />
                Synthesized Operational Answer
              </div>
              <p className="text-sm text-slate-100 leading-relaxed font-normal">
                {result.executive_summary || result.root_cause_analysis?.summary || "No review summary available."}
              </p>
            </div>

            {/* Primary Operational Reasons */}
            {result.key_reasons && result.key_reasons.length > 0 && (
              <div className="space-y-2 pt-1">
                <span className="text-xs font-bold uppercase tracking-wider text-slate-400 block">
                  Primary Operational Reasons Identified in Reviews:
                </span>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                  {result.key_reasons.map((reason, idx) => (
                    <div key={idx} className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950/60 border border-slate-800/80 text-xs text-slate-200 hover:border-emerald-500/30 transition">
                      <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                        {idx + 1}
                      </span>
                      <span className="leading-snug">{reason}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* 2. Customer Review Evidence (Supporting Data) */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-sky-400" />
                <h3 className="font-bold text-white text-base">
                  Customer Review Evidence ({result.retrieved_reviews?.length || 0})
                </h3>
              </div>
              <span className="text-xs text-slate-400">Grounding Evidence Ranked by Operational Relevance</span>
            </div>

            {(!result.retrieved_reviews || result.retrieved_reviews.length === 0) ? (
              <div className="bg-emerald-950/20 border border-emerald-500/30 rounded-xl p-6 text-center space-y-2.5">
                <div className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 mb-1">
                  <CheckCircle className="w-6 h-6" />
                </div>
                <h4 className="font-bold text-sm text-white">
                  {result.target_restaurant && result.target_restaurant !== 'All Restaurants'
                    ? `No Complaints Found for ${result.target_restaurant}`
                    : 'No Matching Complaints Found'}
                </h4>
                <p className="text-xs text-slate-300 max-w-lg mx-auto leading-relaxed">
                  {result.no_complaint_notice ||
                    `Customer reviews for ${result.target_restaurant || 'the selected restaurant'} do not contain reports of this issue. Customer feedback on this operational area is clean and satisfactory.`}
                </p>
                <div className="inline-flex items-center gap-1.5 text-[11px] text-emerald-400 font-semibold bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 mt-1">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  Grounded on Real Yelp Historical Review Evidence
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {result.retrieved_reviews.map((r, i) => (
                  <div key={i} className="bg-slate-950 p-4 rounded-xl border border-slate-800 hover:border-slate-700 transition space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="px-2 py-0.5 rounded bg-sky-500/15 border border-sky-500/30 text-[10px] font-bold text-sky-400">
                          Evidence Case #{i + 1}
                        </span>
                        <span className="font-bold text-xs text-white">{r.restaurant_name}</span>
                        {(r.city || r.state) && (
                          <span className="text-[11px] text-slate-500">
                            {[r.city, r.state].filter(Boolean).join(', ')}
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1 text-amber-400 font-bold text-xs">
                        <Star className="w-3 h-3 fill-amber-400 text-amber-400" />
                        <span>{r.stars}</span>
                      </div>
                    </div>

                    <p className="text-xs text-slate-300 italic line-clamp-3 leading-relaxed">
                      "{r.text}"
                    </p>

                    <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px]">
                      <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400">
                        {r.category || 'Customer Review'}
                      </span>
                      <span className="text-emerald-400 font-bold bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full text-[10px]">
                        {Math.round((r.hybrid_score || 0.85) * 100)}% Relevance Match
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 3. Priority & Classification Banner */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Priority Card */}
            <div className={`p-5 rounded-2xl border ${
              result.prioritization?.action_color === 'red' ? 'bg-rose-950/20 border-rose-500/40' :
              result.prioritization?.action_color === 'orange' ? 'bg-amber-950/20 border-amber-500/40' :
              result.prioritization?.action_color === 'emerald' ? 'bg-emerald-950/20 border-emerald-500/40' :
              'bg-sky-950/20 border-sky-500/40'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold uppercase text-slate-400">Issue Priority</span>
                <span className={`px-2.5 py-0.5 rounded-full text-xs font-black ${
                  result.prioritization?.action_color === 'red' ? 'bg-rose-500/20 text-rose-400 border border-rose-500/30' :
                  result.prioritization?.action_color === 'orange' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                  result.prioritization?.action_color === 'emerald' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                  'bg-sky-500/20 text-sky-400 border border-sky-500/30'
                }`}>
                  {result.prioritization?.priority_tier}
                </span>
              </div>
              <div className="text-2xl font-black text-white mb-1">
                Score: {result.prioritization?.priority_score} <span className="text-xs font-normal text-slate-400">/ 100</span>
              </div>
              <div className="text-xs text-slate-300 mb-2">
                <strong>Recommended SLA:</strong> {result.prioritization?.sla}
              </div>
              {result.prioritization?.critical_hazard_flag && (
                <div className="text-xs text-rose-300 bg-rose-500/20 p-2 rounded-lg border border-rose-500/30 font-semibold flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  {result.prioritization?.critical_hazard_flag}
                </div>
              )}
            </div>

            {/* Classification Card */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-2">Issue Category</span>
              <div className="text-lg font-bold text-emerald-400 mb-1">
                {result.classification?.primary_category}
              </div>
              <div className="text-xs text-slate-300 mb-2">
                <strong>Specific Area:</strong> {result.classification?.sub_category}
              </div>
              <div className="flex items-center justify-between text-xs text-slate-400 border-t border-slate-800 pt-2">
                <span>Confidence: <strong>{(result.classification?.confidence * 100).toFixed(0)}%</strong></span>
                <span>Customer Mood: <strong>{result.classification?.sentiment}</strong></span>
              </div>
            </div>

            {/* Quality Audit Card */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold uppercase text-slate-400">Action Plan Quality</span>
                <span className="text-xs font-bold text-emerald-400">
                  {result.judge_audit?.overall_score_pct}% Approved
                </span>
              </div>
              <div className="text-sm font-bold text-white mb-1">
                {result.judge_audit?.verdict}
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400 mt-2">
                <div>Evidence: <strong className="text-white">{result.judge_audit?.rubric_scores?.evidence_grounding}/5</strong></div>
                <div>Root Cause: <strong className="text-white">{result.judge_audit?.rubric_scores?.root_cause_precision}/5</strong></div>
                <div>Actionability: <strong className="text-white">{result.judge_audit?.rubric_scores?.actionability}/5</strong></div>
                <div>Calibration: <strong className="text-white">{result.judge_audit?.rubric_scores?.severity_calibration}/5</strong></div>
              </div>
            </div>
          </div>

          {/* 4. Operational Recommendations & Preventative SOPs */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Immediate Actions */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-amber-400" />
                <h3 className="font-bold text-white text-base">Immediate Corrective Actions (Next 24h)</h3>
              </div>
              <div className="space-y-2.5">
                {result.recommendations?.immediate_actions?.map((act, i) => (
                  <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-200">
                    <span className="w-5 h-5 rounded-full bg-amber-400/20 text-amber-400 flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      {i + 1}
                    </span>
                    <span className="leading-relaxed">{act}</span>
                  </div>
                ))}
              </div>

              {/* Staff Directives */}
              <div className="pt-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Staff Coaching Directives</h4>
                {result.recommendations?.staff_coaching?.map((coach, i) => (
                  <div key={i} className="p-2.5 rounded-lg bg-slate-950/60 text-xs text-slate-300 border border-slate-800/50 mb-1.5">
                    • {coach}
                  </div>
                ))}
              </div>
            </div>

            {/* Preventative SOPs & KPIs */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-white text-base">Standard Operating Procedures (SOPs)</h3>
              </div>
              <div className="space-y-2.5">
                {result.recommendations?.preventative_sops?.map((sop, i) => (
                  <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-200">
                    <span className="w-5 h-5 rounded-full bg-emerald-400/20 text-emerald-400 flex items-center justify-center font-bold text-[10px] flex-shrink-0 mt-0.5">
                      ✓
                    </span>
                    <span className="leading-relaxed font-mono">{sop}</span>
                  </div>
                ))}
              </div>

              {/* Success KPIs */}
              <div className="pt-2">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Target Performance Metrics</h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {result.recommendations?.success_kpis?.map((kpi, i) => (
                    <div key={i} className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-xs">
                      <div className="text-slate-400 text-[11px]">{kpi.metric}</div>
                      <div className="text-emerald-400 font-bold mt-0.5">{kpi.target}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Manager Feedback Card */}
          <div className="bg-slate-900/90 border border-emerald-500/30 rounded-2xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-white text-base">Manager Feedback</h3>
              </div>
              <span className="text-xs text-emerald-400 font-medium bg-emerald-500/10 px-2.5 py-1 rounded-full border border-emerald-500/20">
                Continuous Learning System
              </span>
            </div>
            <p className="text-xs text-slate-400 mb-4">
              Your feedback is recorded to adapt future recommendations and operational protocols.
            </p>

            {feedbackSubmitted ? (
              <div className="p-4 rounded-xl bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
                <CheckCircle className="w-4 h-4" />
                <span>Feedback recorded successfully! Thank you for helping improve the operational assistant.</span>
              </div>
            ) : (
              <form onSubmit={handleFeedbackSubmit} className="space-y-4">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-300 font-medium">Was this recommendation helpful?</span>
                    <button
                      type="button"
                      onClick={() => setFeedbackHelpful(true)}
                      className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 transition ${
                        feedbackHelpful ? 'bg-emerald-500/20 border-emerald-500 text-emerald-300' : 'bg-slate-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      <ThumbsUp className="w-3.5 h-3.5" /> Yes
                    </button>
                    <button
                      type="button"
                      onClick={() => setFeedbackHelpful(false)}
                      className={`px-3 py-1.5 rounded-lg border flex items-center gap-1.5 transition ${
                        !feedbackHelpful ? 'bg-rose-500/20 border-rose-500 text-rose-300' : 'bg-slate-950 border-slate-800 text-slate-400'
                      }`}
                    >
                      <ThumbsDown className="w-3.5 h-3.5" /> Needs Work
                    </button>
                  </div>

                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-slate-300 font-medium">Rating:</span>
                    {[1, 2, 3, 4, 5].map((num) => (
                      <button
                        key={num}
                        type="button"
                        onClick={() => setFeedbackRating(num)}
                        className={`w-7 h-7 rounded-lg text-xs font-bold border transition ${
                          feedbackRating >= num ? 'bg-amber-500/20 border-amber-500/50 text-amber-300' : 'bg-slate-950 border-slate-800 text-slate-500'
                        }`}
                      >
                        {num}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <input
                    type="text"
                    value={feedbackNotes}
                    onChange={(e) => setFeedbackNotes(e.target.value)}
                    placeholder="Add optional notes (e.g. shift manager notes, table observations)..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-xs text-white placeholder-slate-600 focus:outline-none focus:border-emerald-500"
                  />
                </div>

                <div className="flex justify-end">
                  <button
                    type="submit"
                    className="px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-bold rounded-xl text-xs transition shadow-md shadow-emerald-500/20"
                  >
                    Submit Feedback
                  </button>
                </div>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
