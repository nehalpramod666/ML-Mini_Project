import { useState } from "react";
import { uploadFile } from "../api/client";
import ErrorMessage from "./ErrorMessage";

function UploadForm({ onUploadSuccess, busy = false }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | uploading | error
  const [errorMessage, setErrorMessage] = useState("");

  const uploading = status === "uploading";
  const buttonLabel = uploading ? "Uploading..." : busy ? "Analyzing..." : "Upload";

  const handleFile = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setStatus("idle");
    setErrorMessage("");
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setStatus("uploading");
    setErrorMessage("");

    try {
      const result = await uploadFile(selectedFile);
      setStatus("idle");
      onUploadSuccess(result);
    } catch (err) {
      setStatus("error");
      setErrorMessage(err.message || "Upload failed");
    }
  };

  return (
    <div className="w-full max-w-md flex flex-col gap-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
          isDragging ? "border-blue-400 bg-gray-800" : "border-gray-600"
        }`}
        onClick={() => document.getElementById("fileInput").click()}
      >
        <input
          id="fileInput"
          type="file"
          accept=".dcm"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        <p className="text-gray-300">
          {selectedFile
            ? `Selected: ${selectedFile.name}`
            : "Drag & drop a .dcm file here, or click to select"}
        </p>
      </div>

      <button
        onClick={handleUpload}
        disabled={!selectedFile || uploading || busy}
        className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-2 rounded-lg transition-colors"
      >
        {buttonLabel}
      </button>

      <ErrorMessage message={status === "error" ? errorMessage : ""} />
    </div>
  );
}

export default UploadForm;