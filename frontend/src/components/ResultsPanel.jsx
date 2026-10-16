function stateColor(state) {
  switch (state) {
    case "ACCEPT":
      return "text-green-400";
    case "BORDERLINE":
      return "text-yellow-400";
    case "REJECT":
      return "text-red-400";
    default:
      return "text-gray-400";
  }
}

function MetricBar({ label, value, max = 100 }) {
  const numericValue = typeof value === "number" ? value : 0;
  const percent = Math.max(0, Math.min((numericValue / max) * 100, 100));

  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-gray-400 mb-1">
        <span>{label}</span>
        <span>{value !== undefined && value !== null ? `${Math.round(value)}/100` : "N/A"}</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div
          className="bg-blue-500 h-2 rounded-full transition-all"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

function ResultsPanel({ assessment, result }) {
  const data = assessment || result?.assessment || result;
  if (!data) return null;

  const { metrics, fuzzy_score, state, dominant_defect } = data;

  return (
    <div className="w-full max-w-md bg-gray-800 rounded-lg p-6 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Quality Score</span>
        <span className="text-2xl font-bold">{fuzzy_score}/100</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Decision</span>
        <span className={`font-bold ${stateColor(state)}`}>{state}</span>
      </div>

      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Dominant Defect</span>
        <span className="text-gray-200">{dominant_defect}</span>
      </div>

      <div className="flex flex-col gap-3 mt-2">
        <MetricBar label="Blur severity" value={metrics?.blur} />
        <MetricBar label="Contrast severity" value={metrics?.contrast} />
        <MetricBar label="SNR severity" value={metrics?.noise} />
      </div>
    </div>
  );
}

export default ResultsPanel;