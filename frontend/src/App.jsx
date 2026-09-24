import { useEffect, useState } from "react";
import { checkHealth } from "./api/client";
import UploadForm from "./components/UploadForm";

function App() {
  const [status, setStatus] = useState("checking...");
  const [uploadResult, setUploadResult] = useState(null);

  useEffect(() => {
    checkHealth()
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("backend unreachable"));
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center gap-6 p-8">
      <h1 className="text-3xl font-bold">Clinical Gatekeeper</h1>
      <p className="text-sm text-gray-400">Backend status: {status}</p>

      <UploadForm onUploadSuccess={setUploadResult} />

      {uploadResult && (
        <div className="text-sm text-green-400 mt-4">
          Uploaded! file_id: {uploadResult.file_id}
        </div>
      )}
    </div>
  );
}

export default App;