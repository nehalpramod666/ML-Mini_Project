import { useEffect, useRef, useState } from "react";
import { RenderingEngine, Enums, init as coreInit } from "@cornerstonejs/core";
import { init as dicomImageLoaderInit } from "@cornerstonejs/dicom-image-loader";
import { registerDefaultProviders } from "@cornerstonejs/metadata";
const BASE_URL = "http://localhost:8000";

// Cornerstone must be initialised once per page load, not once per render.
let initPromise = null;
function initCornerstone() {
  if (!initPromise) {
    initPromise = (async () => {
      await coreInit();
      await registerDefaultProviders();
      await dicomImageLoaderInit();
    })().catch((err) => {
      initPromise = null; // allow a retry after a failure
      throw err;
    });
  }
  return initPromise;
}

function DicomViewport({ fileId }) {
  const elementRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!fileId) return undefined;

    let cancelled = false;
    let renderingEngine = null;
    setLoading(true);
    setError("");

    (async () => {
      try {
        await initCornerstone();
        if (cancelled || !elementRef.current) return;

        renderingEngine = new RenderingEngine(`engine-${fileId}`);
        const viewportId = "dicom-viewport";

        renderingEngine.enableElement({
          viewportId,
          type: Enums.ViewportType.STACK,
          element: elementRef.current,
        });

        const viewport = renderingEngine.getViewport(viewportId);
        await viewport.setStack([`wadouri:${BASE_URL}/files/${fileId}`], 0);
        
        if (cancelled) return;

        viewport.render();
        setLoading(false);
      } catch (err) {
        if (cancelled) return;
        console.error("DICOM viewer error:", err);
        setError("The image could not be displayed.");
        setLoading(false);
      }
    })();

    return () => {
      cancelled = true;
      if (renderingEngine) renderingEngine.destroy();
    };
  }, [fileId]);

  return (
    <div className="relative w-full h-96 bg-black rounded-lg overflow-hidden">
      <div
        ref={elementRef}
        onContextMenu={(e) => e.preventDefault()}
        className="w-full h-full"
      />
      {loading && !error && (
        <p className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
          Loading image...
        </p>
      )}
      {error && (
        <p className="absolute inset-0 flex items-center justify-center text-sm text-red-300 px-4 text-center">
          {error}
        </p>
      )}
    </div>
  );
}

export default DicomViewport;