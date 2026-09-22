// Centralized API client — all backend calls go through here.
// Base URL points to FastAPI backend (localhost:8000 during dev).

const BASE_URL = "http://localhost:8000";

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}