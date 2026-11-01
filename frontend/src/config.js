// Single place for frontend configuration.
// Override with VITE_API_BASE_URL (for example in a .env file); otherwise
// the local backend is used.
export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";