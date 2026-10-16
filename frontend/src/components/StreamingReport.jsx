import { useEffect, useState } from "react";

function useTypewriter(text, active) {
  const [displayedText, setDisplayedText] = useState("");
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (!active || !text) {
      setDisplayedText(text || "");
      setIsAnimating(false);
      return;
    }

    setDisplayedText("");
    setIsAnimating(true);

    const words = text.split(" ");
    let index = 0;

    const interval = setInterval(() => {
      index++;
      setDisplayedText(words.slice(0, index).join(" "));

      if (index >= words.length) {
        clearInterval(interval);
        setIsAnimating(false);
      }
    }, 60);

    return () => clearInterval(interval);
  }, [text, active]);

  return { displayedText, isAnimating };
}

function StreamingReport({ report, active = true }) {
  const { displayedText, isAnimating } = useTypewriter(report, active);

  if (!active || !report) return null;

  return (
    <div className="w-full max-w-md bg-gray-800 rounded-lg p-6">
      <div className="flex items-center justify-between mb-2">
        <span className="text-gray-400 text-sm">Clinical Instruction</span>
        {isAnimating && (
          <span className="text-xs text-blue-400 animate-pulse">
            rendering...
          </span>
        )}
      </div>
      <p className="text-gray-100 leading-relaxed">
        {displayedText}
        {isAnimating && <span className="animate-pulse">▌</span>}
      </p>
    </div>
  );
}

export default StreamingReport;