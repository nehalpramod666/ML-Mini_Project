import { Link, Navigate } from "react-router-dom";
import ResultsPanel from "../components/ResultsPanel";
import StreamingReport from "../components/StreamingReport";

function ResultsPage({ result, onReset }) {
  // No result yet (e.g. page refreshed or URL typed directly): go back to upload.
  if (!result) return <Navigate to="/" replace />;

  return (
    <div className="w-full max-w-md flex flex-col items-center gap-6">
      <ResultsPanel result={result} />
      <StreamingReport active={true} />
      <Link
        to="/"
        onClick={onReset}
        className="text-blue-400 hover:text-blue-300 text-sm"
      >
        ← Analyze another file
      </Link>
    </div>
  );
}

export default ResultsPage;