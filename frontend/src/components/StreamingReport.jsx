import { useEffect, useState } from "react";

// TEMPORARY fake token source — simulates the real LLM stream.
// Week 7 will replace this with a real SSE/WebSocket connection to
// the backend's /ws/report endpoint. The component's external
// behavior (props in, rendered text out) stays the same either way.
const FAKE_REPORT =
  "Image rejected due to severe motion artifact. Advise patient to suspend respiration for the next capture attempt. Reposition the detector and confirm patient immobilization before retaking the exposure.";

function useFakeTokenStream(active) {
  const [displayedText, setDisplayedText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);

  useEffect(() => {
    if (!active) {
      setDisplayedText("");
      setIsStreaming(false);
      return;
    }

    setDisplayedText("");
    setIsStreaming(true);

    const words = FAKE_REPORT.split(" ");
    let index = 0;

    const interval = setInterval(() => {
      index++;
      setDisplayedText(words.slice(0, index).join(" "));

      if (index >= words.length) {
        clearInterval(interval);
        setIsStreaming(false);
      }
    }, 80); // ms per word — tune for a natural typewriter feel

    return () => clearInterval(interval);
  }, [active]);

  return { displayedText, isStreaming };
}

function StreamingReport({ active }) {
  const { displayedText, isStreaming } = useFakeTokenStream(active);

  if (!active) return null;

  return (
    <div className="w-full max-w-md bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">Clinical Instruction</span>
        {isStreaming && (
          <span className="text-xs text-blue-400 animate-pulse">
            generating...
          </span>
        )}
      </div>
      <p className="text-gray-100 leading-relaxed">
        {displayedText}
        {isStreaming && <span className="animate-pulse">▌</span>}
      </p>
    </div>
  );
}

export default StreamingReport;