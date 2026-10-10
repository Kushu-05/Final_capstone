function getApiBaseUrl() {
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname;
    // In local development, connect directly to FastAPI backend on port 8000
    // to completely prevent any dev server proxy socket restrictions
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return `http://${hostname}:8000/api`;
    }
  }
  return '/api';
}

const BASE_URL = getApiBaseUrl();

export async function fetchHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

export async function fetchConfig() {
  const res = await fetch(`${BASE_URL}/config`);
  return res.json();
}

export async function updateConfig(data) {
  const res = await fetch(`${BASE_URL}/config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  return res.json();
}

export async function fetchMetadata() {
  const res = await fetch(`${BASE_URL}/metadata`);
  return res.json();
}

export async function runMultiAgent(payload) {
  const res = await fetch(`${BASE_URL}/multi-agent`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Multi-agent execution error');
  }
  return res.json();
}

export async function runSearch(payload) {
  const res = await fetch(`${BASE_URL}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || 'Search failed');
  }
  return res.json();
}

export async function predictCSAT(text) {
  const res = await fetch(`${BASE_URL}/csat-predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function fetchTrends(payload) {
  const res = await fetch(`${BASE_URL}/trends`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  return res.json();
}

export async function runEvaluation(payload) {
  const res = await fetch(`${BASE_URL}/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function submitFeedback(payload) {
  const res = await fetch(`${BASE_URL}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function fetchFeedback() {
  const res = await fetch(`${BASE_URL}/feedback`);
  return res.json();
}
