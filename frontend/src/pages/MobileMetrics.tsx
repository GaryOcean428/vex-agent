import { useNavigate } from "react-router-dom";
import { useMetricsHistory, useVexState } from "../hooks/index.ts";
import { MetricsSidebar } from "../components/chat/MetricsSidebar.tsx";

/**
 * Full-screen metrics page for mobile viewports.
 * Renders the same MetricsSidebar content in a full-width layout
 * with a back button to return to the previous page.
 */
export default function MobileMetrics() {
  const navigate = useNavigate();
  const { data: state } = useVexState();
  const history = useMetricsHistory(state, 60);

  return (
    <div className="mobile-metrics-page">
      <header className="mobile-metrics-header">
        <button
          className="mobile-metrics-back"
          onClick={() => window.history.length > 1 ? navigate(-1) : navigate('/chat')}
          aria-label="Go back"
        >
          &larr; Back
        </button>
        <h1 className="mobile-metrics-title">Metrics</h1>
      </header>

      <div className="mobile-metrics-content">
        <MetricsSidebar
          state={state ?? null}
          history={history}
          kernelSummary={state?.kernels ?? null}
          emotion={null}
          precog={null}
          learning={null}
          visible={true}
          standalone
        />
      </div>
    </div>
  );
}
