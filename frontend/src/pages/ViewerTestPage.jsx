import { useParams } from "react-router-dom";
import DicomViewport from "../components/DicomViewport";

function ViewerTestPage() {
  const { fileId } = useParams();
  return (
    <div className="w-full max-w-xl">
      <DicomViewport fileId={fileId} />
    </div>
  );
}

export default ViewerTestPage;