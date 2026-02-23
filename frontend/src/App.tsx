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

// Lazy-loaded dashboard pages
const Overview = lazy(() => import("./pages/dashboard/Overview.tsx"));
const Consciousness = lazy(() => import("./pages/dashboard/Consciousness.tsx"));
const Basins = lazy(() => import("./pages/dashboard/Basins.tsx"));
const Graph = lazy(() => import("./pages/dashboard/Graph.tsx"));
const Lifecycle = lazy(() => import("./pages/dashboard/Lifecycle.tsx"));
const Cognition = lazy(() => import("./pages/dashboard/Cognition.tsx"));
const Memory = lazy(() => import("./pages/dashboard/Memory.tsx"));
const Telemetry = lazy(() => import("./pages/dashboard/Telemetry.tsx"));
const Training = lazy(() => import("./pages/dashboard/Training.tsx"));
const Governor = lazy(() => import("./pages/dashboard/Governor.tsx"));
const Admin = lazy(() => import("./pages/dashboard/Admin.tsx"));

const PageLoading = () => <div className="page-loading">Loading...</div>;

const router = createBrowserRouter([
  {
    path: "/login",
    element: <Login />,
  },
  {
    element: <ProtectedRoute />,
    children: [
      {
        element: <Layout />,
        children: [
          { path: "/", element: <Navigate to="/chat" replace /> },
          { path: "/chat", element: <Suspense fallback={<PageLoading />}><Chat /></Suspense> },
          { path: "/chat/:conversationId", element: <Suspense fallback={<PageLoading />}><Chat /></Suspense> },
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
  return (
    <ErrorBoundary>
      <AuthProvider>
        <RouterProvider router={router} />
        <ToastContainer />
      </AuthProvider>
    </ErrorBoundary>
  );
}
