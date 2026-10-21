function Spinner({ label }) {
  return (
    <div role="status" className="flex items-center gap-2 text-gray-400 text-sm">
      <span className="h-4 w-4 rounded-full border-2 border-gray-500 border-t-transparent animate-spin" />
      <span>{label}</span>
    </div>
  );
}

export default Spinner;