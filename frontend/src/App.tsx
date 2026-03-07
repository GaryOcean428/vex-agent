import { lazy, Suspense } from "react";
import {
    createBrowserRouter,
    Navigate,
    RouterProvider,
} from "react-router-dom";
import { AuthProvider } from "./auth/AuthContext.tsx";
import ProtectedRoute from "./auth/ProtectedRoute.tsx";
import { ErrorBoundary } from "./components/ErrorBoundary.tsx";
import Layout from "./components/Layout.tsx";
import { ToastContainer } from "./components/Toast.tsx";
import Chat from "./pages/Chat.tsx";
import Dashboard from "./pages/dashboard/Dashboard.tsx";
import Login from "./pages/Login.tsx";

// Retry dynamic imports once then force-reload on stale chunk errors
function lazyRetry<T extends React.ComponentType>(
  factory: () => Promise<{ default: T }>,
) {
  return lazy(() =>
    factory().catch((err: unknown) => {
      const alreadyRetried = sessionStorage.getItem("chunk-retry");
      if (!alreadyRetried) {
        sessionStorage.setItem("chunk-retry", "1");
        window.location.reload();
      }
      throw err;
    }),
  );
}

// Lazy-loaded dashboard pages
const Overview = lazyRetry(() => import("./pages/dashboard/Overview.tsx"));
const Consciousness = lazyRetry(() => import("./pages/dashboard/Consciousness.tsx"));
const Basins = lazyRetry(() => import("./pages/dashboard/Basins.tsx"));
const Graph = lazyRetry(() => import("./pages/dashboard/Graph.tsx"));
const Lifecycle = lazyRetry(() => import("./pages/dashboard/Lifecycle.tsx"));
const Cognition = lazyRetry(() => import("./pages/dashboard/Cognition.tsx"));
const Memory = lazyRetry(() => import("./pages/dashboard/Memory.tsx"));
const Telemetry = lazyRetry(() => import("./pages/dashboard/Telemetry.tsx"));
const Training = lazyRetry(() => import("./pages/dashboard/Training.tsx"));
const Governor = lazyRetry(() => import("./pages/dashboard/Governor.tsx"));
const Admin = lazyRetry(() => import("./pages/dashboard/Admin.tsx"));
const MobileMetrics = lazyRetry(() => import("./pages/MobileMetrics.tsx"));

const PageLoading = () => <div className="page-loading">Loading...</div>;

function RouteError() {
  return (
    <div className="error-boundary">
      <div className="error-boundary-card">
        <div className="error-boundary-icon" aria-hidden="true">!</div>
        <h1 className="error-boundary-title">Page failed to load</h1>
        <p className="error-boundary-message">
          This usually means a new version was deployed. Reloading should fix it.
        </p>
        <div className="error-boundary-actions">
          <button
            type="button"
            className="error-boundary-btn error-boundary-btn-primary"
            onClick={() => {
              sessionStorage.removeItem("chunk-retry");
              window.location.reload();
            }}
          >
            Reload
          </button>
          <button
            type="button"
            className="error-boundary-btn error-boundary-btn-secondary"
            onClick={() => { window.location.href = "/chat"; }}
          >
            Go Home
          </button>
        </div>
      </div>
    </div>
  );
}

const router = createBrowserRouter([
  {
    path: "/login",
    element: <Login />,
  },
  {
    element: <ProtectedRoute />,
    errorElement: <RouteError />,
    children: [
      {
        element: <Layout />,
        errorElement: <RouteError />,
        children: [
          { path: "/", element: <Navigate to="/chat" replace /> },
          { path: "/chat", element: <Suspense fallback={<PageLoading />}><Chat /></Suspense> },
          { path: "/chat/:conversationId", element: <Suspense fallback={<PageLoading />}><Chat /></Suspense> },
          { path: "/metrics", element: <Suspense fallback={<PageLoading />}><MobileMetrics /></Suspense> },
          {
            path: "/dashboard",
            element: <Dashboard />,
            children: [
              { index: true, element: <Suspense fallback={<PageLoading />}><Overview /></Suspense> },
              { path: "consciousness", element: <Suspense fallback={<PageLoading />}><Consciousness /></Suspense> },
              { path: "basins", element: <Suspense fallback={<PageLoading />}><Basins /></Suspense> },
              { path: "graph", element: <Suspense fallback={<PageLoading />}><Graph /></Suspense> },
              { path: "lifecycle", element: <Suspense fallback={<PageLoading />}><Lifecycle /></Suspense> },
              { path: "cognition", element: <Suspense fallback={<PageLoading />}><Cognition /></Suspense> },
              { path: "memory", element: <Suspense fallback={<PageLoading />}><Memory /></Suspense> },
              { path: "telemetry", element: <Suspense fallback={<PageLoading />}><Telemetry /></Suspense> },
              { path: "training", element: <Suspense fallback={<PageLoading />}><Training /></Suspense> },
              { path: "governor", element: <Suspense fallback={<PageLoading />}><Governor /></Suspense> },
              { path: "admin", element: <Suspense fallback={<PageLoading />}><Admin /></Suspense> },
            ],
          },
          { path: "*", element: <Navigate to="/chat" replace /> },
        ],
      },
    ],
  },
]);

export default function App() {
  // Clear stale-chunk retry flag on successful load
  sessionStorage.removeItem("chunk-retry");

  return (
    <ErrorBoundary>
      <AuthProvider>
        <RouterProvider router={router} />
        <ToastContainer />
      </AuthProvider>
    </ErrorBoundary>
  );
}
