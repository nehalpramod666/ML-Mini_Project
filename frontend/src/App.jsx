import { useEffect, useState } from "react";
import { checkHealth } from "./api/client";

function App() {
  const [status, setStatus] = useState("checking...");

  useEffect(() => {
    checkHealth()
      .then((data) => setStatus(data.status))
      .catch(() => setStatus("backend unreachable"));
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center gap-4">
      <h1 className="text-3xl font-bold">Hello Gatekeeper</h1>
      <p className="text-sm text-gray-400">Backend status: {status}</p>
    </div>
  );
}

export default App;