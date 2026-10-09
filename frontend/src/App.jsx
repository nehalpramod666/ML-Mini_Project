import { useEffect, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { checkHealth } from "./api/client";
import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";

function App() {
  const [status, setStatus] = useState("checking...");
  const [analysisResult, setAnalysisResult] = useState(null);

  useEffect(() => {
    checkHealth()
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("backend unreachable"));
  }, []);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center gap-6 p-8">
        <h1 className="text-3xl font-bold">Clinical Gatekeeper</h1>
        <p className="text-sm text-gray-400">Backend status: {status}</p>

        <Routes>
          <Route
            path="/"
            element={<UploadPage onResult={setAnalysisResult} />}
          />
          <Route
            path="/results"
            element={
              <ResultsPage
                result={analysisResult}
                onReset={() => setAnalysisResult(null)}
              />
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;