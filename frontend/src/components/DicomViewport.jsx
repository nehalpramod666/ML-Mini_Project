import { useEffect, useRef, useState } from "react";
import { RenderingEngine, Enums, init as coreInit } from "@cornerstonejs/core";
import { registerDefaultProviders } from "@cornerstonejs/metadata";
import { init as dicomImageLoaderInit } from "@cornerstonejs/dicom-image-loader";
import {
  init as cornerstoneToolsInit,
  addTool,
  ToolGroupManager,
  WindowLevelTool,
  Enums as csToolsEnums,
} from "@cornerstonejs/tools";

const BASE_URL = "http://localhost:8000";

// Cornerstone must be initialised once per page load, not once per render.
let initPromise = null;
function initCornerstone() {
  if (!initPromise) {
    initPromise = (async () => {
      await coreInit();
      await registerDefaultProviders();
      await dicomImageLoaderInit();
      await cornerstoneToolsInit();
      try {
        addTool(WindowLevelTool);
      } catch {
        // Already registered (for example after a hot reload); safe to ignore.
      }
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
    let toolGroupId = null;
    setLoading(true);
    setError("");

    (async () => {
      try {
        await initCornerstone();
        if (cancelled || !elementRef.current) return;

        const renderingEngineId = `engine-${fileId}`;
        const viewportId = "dicom-viewport";
        toolGroupId = `toolgroup-${fileId}`;

        renderingEngine = new RenderingEngine(renderingEngineId);
        renderingEngine.enableElement({
          viewportId,
          type: Enums.ViewportType.STACK,
          element: elementRef.current,
        });

        // Window/level (brightness and contrast) on the left mouse button.
        const toolGroup = ToolGroupManager.createToolGroup(toolGroupId);
        toolGroup.addTool(WindowLevelTool.toolName);
        toolGroup.addViewport(viewportId, renderingEngineId);
        toolGroup.setToolActive(WindowLevelTool.toolName, {
          bindings: [{ mouseButton: csToolsEnums.MouseBindings.Primary }],
        });

        const viewport = renderingEngine.getViewport(viewportId);
        await viewport.setStack([`wadouri:${BASE_URL}/files/${fileId}`], 0);
        if (cancelled) return;
        viewport.render();

        // setStack does not reject when the image fails to load, so check
        // that the viewport really holds image data.
        const hasImage =
          typeof viewport.getImageData === "function"
            ? Boolean(viewport.getImageData())
            : true;
        if (!hasImage) {
          throw new Error("The viewport has no image data after loading.");
        }

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
      if (toolGroupId) ToolGroupManager.destroyToolGroup(toolGroupId);
      if (renderingEngine) renderingEngine.destroy();
    };
  }, [fileId]);

  return (
    <div className="w-full">
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
      <p className="mt-2 text-xs text-gray-400 text-center">
        Drag with the left mouse button to adjust brightness and contrast.
      </p>
    </div>
  );
}

export default DicomViewport;