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

export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/upload`, {
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
  const res = await fetch(`${BASE_URL}/analyze/${fileId}`, {
    method: "POST",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(errorData, "Analysis failed"));
  }

  return res.json();
}