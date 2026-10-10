import React, { useState } from 'react';
import { BarChart3, Star, ThumbsUp, ThumbsDown, Zap, Sparkles, CheckCircle, AlertOctagon } from 'lucide-react';
import { predictCSAT } from '../api';

const SAMPLE_TEXTS = [
  "We waited 50 minutes for cold pasta and our server completely ignored our table. Terrible experience.",
  "Found a long hair in our pizza and the manager was dismissive. Will never return.",
  "Decent burger but the fries were a bit soggy and the music was too loud to talk.",
  "Outstanding dining! Hand-made ravioli was heavenly and the service was warm and prompt."
];

export default function CSATPredictorView() {
  const [text, setText] = useState(SAMPLE_TEXTS[0]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (t) => {
    const textToRun = t || text;
    if (!textToRun.trim()) return;

    setLoading(true);
    try {
      const res = await predictCSAT(textToRun);
      setPrediction(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
        <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-1">
          <BarChart3 className="w-5 h-5 text-purple-400" />
          Customer Satisfaction (CSAT) & Rating Predictor
        </h2>
        <p className="text-xs text-slate-400 mb-5">
          Machine learning regression model trained on historical Yelp customer sentiment and linguistic review patterns.
        </p>

        {/* Input */}
        <div className="space-y-3">
          <textarea
            rows={3}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Type or paste a customer complaint or review..."
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 resize-none"
          />

          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] text-slate-500 font-semibold uppercase">Presets:</span>
              {SAMPLE_TEXTS.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setText(s);
                    handlePredict(s);
                  }}
                  className="text-[11px] text-slate-300 hover:text-white bg-slate-800/60 hover:bg-slate-800 px-2.5 py-1 rounded-lg border border-slate-700/50 transition truncate max-w-xs"
                >
                  Scenario {idx + 1}
                </button>
              ))}
            </div>

            <button
              onClick={() => handlePredict()}
              disabled={loading}
              className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-xl text-xs transition flex items-center gap-2 shadow-lg shadow-purple-600/20 disabled:opacity-50"
            >
              {loading ? 'Evaluating Model...' : 'Predict CSAT'}
            </button>
          </div>
        </div>
      </div>

      {/* Prediction Output */}
      {prediction && (
        <div className="space-y-6 animate-in fade-in duration-300">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Predicted Stars */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Predicted Star Rating</span>
              <div className="flex items-center gap-2">
                <Star className="w-6 h-6 fill-amber-400 text-amber-400" />
                <span className="text-3xl font-black text-white font-mono">
                  {prediction.predicted_stars}
                </span>
                <span className="text-xs text-slate-500">/ 5.0</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                L2 Ridge Regression Output
              </div>
            </div>

            {/* CSAT % */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Satisfaction Index (CSAT)</span>
              <div className="text-3xl font-black text-white font-mono">
                {prediction.predicted_csat_pct}%
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Standardized Customer Score
              </div>
            </div>

            {/* Satisfaction Tier */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">Status Tier</span>
              <div className={`text-xl font-black ${
                prediction.satisfaction_tier === 'Delighted' ? 'text-emerald-400' :
                prediction.satisfaction_tier === 'Satisfied' ? 'text-sky-400' :
                prediction.satisfaction_tier === 'At Risk' ? 'text-amber-400' : 'text-rose-400'
              }`}>
                {prediction.satisfaction_tier}
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Hospitality Health Indicator
              </div>
            </div>

            {/* NPS Category */}
            <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
              <span className="text-xs font-bold uppercase text-slate-400 block mb-1">NPS Classification</span>
              <div className="text-xl font-bold text-white flex items-center gap-1.5">
                <span className={`px-2 py-0.5 rounded-lg text-xs font-black ${
                  prediction.nps_category === 'Promoter' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                  prediction.nps_category === 'Passive' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                  'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                }`}>
                  {prediction.nps_category}
                </span>
                <span className="text-xs font-mono text-slate-400">({prediction.nps_proxy_score > 0 ? `+${prediction.nps_proxy_score}` : prediction.nps_proxy_score})</span>
              </div>
              <div className="text-[11px] text-slate-400 mt-1">
                Net Promoter Score Proxy
              </div>
            </div>
          </div>

          {/* Sentiment Drivers Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Negative Drag Drivers */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
              <div className="flex items-center gap-2">
                <ThumbsDown className="w-5 h-5 text-rose-400" />
                <h3 className="font-bold text-sm text-white uppercase tracking-wider">
                  Negative Sentiment Drag Drivers
                </h3>
              </div>
              <div className="space-y-2">
                {prediction.key_negative_drivers?.map((driver, i) => (
                  <div key={i} className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-rose-300 font-mono flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-rose-400"></span>
                    <span>{driver}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Positive Lift Drivers */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
              <div className="flex items-center gap-2">
                <ThumbsUp className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-sm text-white uppercase tracking-wider">
                  Positive Sentiment Lift Drivers
                </h3>
              </div>
              <div className="space-y-2">
                {prediction.key_positive_drivers?.map((driver, i) => (
                  <div key={i} className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-emerald-300 font-mono flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                    <span>{driver}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Architecture Information */}
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 text-xs text-slate-400 flex items-center justify-between">
            <span>ML Architecture: <strong className="text-slate-300 font-mono">{prediction.model_architecture}</strong></span>
            <span className="text-emerald-400 font-semibold flex items-center gap-1">
              <CheckCircle className="w-3.5 h-3.5" /> Trained Dynamically on Dataset
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
