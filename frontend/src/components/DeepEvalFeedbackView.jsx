import React, { useState, useEffect } from 'react';
import { Award, CheckCircle, AlertCircle, RefreshCw, MessageSquare, ThumbsUp, Database, Sliders } from 'lucide-react';
import { runEvaluation, fetchFeedback } from '../api';

export default function DeepEvalFeedbackView() {
  const [query, setQuery] = useState("Kitchen ticket delays and undercooked chicken entrees");
  const [evalResult, setEvalResult] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [feedbackData, setFeedbackData] = useState(null);
  const [feedbackLoading, setFeedbackLoading] = useState(true);

  const loadFeedbackMetrics = async () => {
    setFeedbackLoading(true);
    try {
      const data = await fetchFeedback();
      setFeedbackData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setFeedbackLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    setEvalLoading(true);
    try {
      const data = await runEvaluation({
        query,
        alpha: 0.5
      });
      setEvalResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setEvalLoading(false);
    }
  };

  useEffect(() => {
    loadFeedbackMetrics();
    handleRunEvaluation();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-1">
          <Award className="w-5 h-5 text-rose-400" />
          DeepEval Recommendation Quality & Feedback Loop
        </h2>
        <p className="text-xs text-slate-400 mb-5">
          Automated evaluation of recommendation faithfulness, relevancy, and feasibility alongside SQLite continuous improvement tracking.
        </p>

        {/* Evaluation Trigger Bar */}
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Scenario to evaluate with DeepEval..."
            className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-rose-500"
          />
          <button
            onClick={handleRunEvaluation}
            disabled={evalLoading}
            className="px-5 py-2.5 bg-rose-500 hover:bg-rose-400 text-slate-950 font-bold rounded-xl text-xs transition flex items-center gap-2 disabled:opacity-50 shadow-md shadow-rose-500/20"
          >
            {evalLoading ? (
              <span className="flex items-center gap-1.5">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Auditing...
              </span>
            ) : (
              'Run DeepEval Audit'
            )}
          </button>
        </div>
      </div>

      {/* DeepEval Test Results Matrix */}
      {evalResult && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5 animate-in fade-in duration-300">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
            <div>
              <span className="text-xs font-bold text-rose-400 uppercase tracking-wider">Evaluation Test Suite</span>
              <h3 className="text-base font-bold text-white mt-0.5">
                DeepEval Metric Verification Results
              </h3>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-xs text-slate-400">Composite Score</div>
                <div className="text-xl font-black text-emerald-400 font-mono">
                  {evalResult.deepeval_suite?.overall_score_pct}%
                </div>
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-black border ${
                evalResult.deepeval_suite?.all_tests_passed
                  ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                  : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
              }`}>
                {evalResult.deepeval_suite?.status}
              </span>
            </div>
          </div>

          {/* Test Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {evalResult.deepeval_suite?.test_results?.map((test, idx) => (
              <div key={idx} className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">{test.metric}</span>
                  {test.passed ? (
                    <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" /> PASS
                    </span>
                  ) : (
                    <span className="text-[10px] font-bold text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded-full border border-rose-500/20 flex items-center gap-1">
                      <AlertCircle className="w-3 h-3" /> FAIL
                    </span>
                  )}
                </div>

                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-black text-white font-mono">{test.score}</span>
                  <span className="text-xs text-slate-500 font-mono">/ 1.0 (min {test.threshold})</span>
                </div>

                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {test.reasoning}
                </p>
              </div>
            ))}
          </div>

          {/* LLM-as-Judge Comparison Card */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <div className="text-xs font-bold text-slate-300 mb-0.5">Complementary LLM-as-Judge Verdict</div>
              <div className="text-xs text-slate-400 font-mono">
                {evalResult.llm_judge_audit?.judge_critique}
              </div>
            </div>
            <div className="flex-shrink-0 text-right">
              <span className="px-3 py-1 bg-slate-900 border border-slate-700 rounded-lg text-xs font-bold text-emerald-400">
                {evalResult.llm_judge_audit?.verdict}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Continuous Improvement Feedback Loop Tracker */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div>
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Database className="w-5 h-5 text-emerald-400" />
              Continuous Learning Feedback Loop (SQLite Database)
            </h3>
            <p className="text-xs text-slate-400">
              Live telemetry of restaurant manager interactions and real-time prompt exemplar adjustments.
            </p>
          </div>
          <button
            onClick={loadFeedbackMetrics}
            className="p-2 text-slate-400 hover:text-white bg-slate-950 rounded-lg border border-slate-800 transition"
            title="Refresh Feedback Logs"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Feedback Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Total Manager Feedback</span>
            <div className="text-2xl font-black text-white font-mono">
              {feedbackData?.metrics?.total_feedback_submissions || 0}
            </div>
            <span className="text-[10px] text-slate-500">Stored in manager_feedback.db</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Manager Approval Rate</span>
            <div className="text-2xl font-black text-emerald-400 font-mono">
              {feedbackData?.metrics?.manager_approval_rate_pct || 100}%
            </div>
            <span className="text-[10px] text-slate-500">Positive helpfulness ratio</span>
          </div>

          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Average Manager Rating</span>
            <div className="text-2xl font-black text-amber-400 font-mono">
              ★ {feedbackData?.metrics?.average_manager_rating || 5.0} <span className="text-xs font-normal text-slate-500">/ 5.0</span>
            </div>
            <span className="text-[10px] text-slate-500">Operational relevance score</span>
          </div>
        </div>

        {/* Feedback Logs Table */}
        <div className="space-y-2">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-400">Recent Manager Feedback Submissions</h4>
          {feedbackData?.recent_feedback?.length === 0 ? (
            <div className="p-8 text-center text-xs text-slate-500 bg-slate-950 rounded-xl border border-slate-800">
              No feedback entries yet. Submit feedback in the Multi-Agent Copilot to populate this live learning loop.
            </div>
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {feedbackData?.recent_feedback?.map((fb, idx) => (
                <div key={idx} className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 text-xs space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-white">Scenario: "{fb.query}"</span>
                    <span className="text-amber-400 font-bold">★ {fb.rating}/5</span>
                  </div>
                  <div className="text-[11px] text-slate-400 flex items-center gap-3">
                    <span>Helpful: <strong>{fb.is_helpful ? 'Yes' : 'No'}</strong></span>
                    <span>Category: <strong>{fb.category}</strong></span>
                    <span>Date: <strong>{fb.created_at?.slice(0, 16)}</strong></span>
                  </div>
                  {fb.manager_notes && (
                    <div className="text-[11px] text-slate-300 italic bg-slate-900/60 p-1.5 rounded border border-slate-800">
                      Notes: {fb.manager_notes}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
