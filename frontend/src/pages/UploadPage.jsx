import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyzeFile } from "../api/client";
import UploadForm from "../components/UploadForm";
import Spinner from "../components/Spinner";
import ErrorMessage from "../components/ErrorMessage";

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
      <UploadForm onUploadSuccess={handleUploadSuccess} busy={analyzing} />
      {analyzing && <Spinner label="Analyzing image..." />}
      <ErrorMessage message={error} />
    </div>
  );
}

export default UploadPage;