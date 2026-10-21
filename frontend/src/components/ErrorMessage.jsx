function ErrorMessage({ message }) {
  if (!message) return null;

  return (
    <div
      role="alert"
      className="w-full max-w-md rounded-lg border border-red-500 bg-gray-800 p-3 text-sm text-red-300"
    >
      {message}
    </div>
  );
}

export default ErrorMessage;