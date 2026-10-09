import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeFile } from "../api/client";
import UploadForm from "../components/UploadForm";

function UploadPage({ onResult }) {
  const navigate = useNavigate();
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  const handleUploadSuccess = async (uploadResult) => {
    setAnalyzing(true);
    setError("");

    try {
      const result = await analyzeFile(uploadResult.file_id);
      onResult(result);
      navigate("/results");
    } catch (err) {
      setError(err.message || "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="w-full max-w-md flex flex-col items-center gap-4">
      <UploadForm onUploadSuccess={handleUploadSuccess} />
      {analyzing && <p className="text-gray-400 text-sm">Analyzing...</p>}
      {error && <p className="text-red-400 text-sm">{error}</p>}
    </div>
  );
}

export default UploadPage;