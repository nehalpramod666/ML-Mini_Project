const BASE_URL = "http://localhost:8000";

function extractErrorMessage(errorData, fallback) {
  if (!errorData || !errorData.detail) {
    return fallback;
  }
  if (typeof errorData.detail === "string") {
    return errorData.detail;
  }
  if (typeof errorData.detail === "object" && errorData.detail !== null) {
    if (errorData.detail.message) {
      return errorData.detail.message;
    }
  }
  return fallback;
}

const NETWORK_ERROR =
  "Cannot reach the backend. Check that the server is running and try again.";

// Wraps fetch so a stopped backend gives a readable message.
async function request(url, options) {
  try {
    return await fetch(url, options);
  } catch {
    throw new Error(NETWORK_ERROR);
  }
}

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await request(`${BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(errorData, "Upload failed"));
  }

  return res.json();
}

export async function analyzeFile(fileId) {
  const res = await request(`${BASE_URL}/analyze/${fileId}`, {
    method: "POST",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(errorData, "Analysis failed"));
  }

  return res.json();
}