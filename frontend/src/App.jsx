import { useEffect, useState } from "react";
import { checkHealth, analyzeFile } from "./api/client";
import UploadForm from "./components/UploadForm";
import ResultsPanel from "./components/ResultsPanel";

function App() {
  const [status, setStatus] = useState("checking...");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    checkHealth()
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("backend unreachable"));
  }, []);

  const handleUploadSuccess = async (uploadResult) => {
    setAnalyzing(true);
    setError("");
    setAnalysisResult(null);

    try {
      const result = await analyzeFile(uploadResult.file_id);
      setAnalysisResult(result);
    } catch (err) {
      setError(err.message || "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-bold">Clinical Gatekeeper</h1>
      <p className="text-sm text-gray-400">Backend status: {status}</p>

      <UploadForm onUploadSuccess={handleUploadSuccess} />

      {analyzing && <p className="text-gray-400 text-sm">Analyzing...</p>}
      {error && <p className="text-red-400 text-sm">{error}</p>}

      <ResultsPanel result={analysisResult} />
    </div>
  );
}

export default App;