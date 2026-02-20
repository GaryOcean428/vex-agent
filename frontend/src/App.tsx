import { createBrowserRouter, RouterProvider, Navigate } from 'react-router-dom';
import { AuthProvider } from './auth/AuthContext.tsx';
import ProtectedRoute from './auth/ProtectedRoute.tsx';
import Layout from './components/Layout.tsx';
import Login from './pages/Login.tsx';
import Chat from './pages/Chat.tsx';
import Dashboard from './pages/dashboard/Dashboard.tsx';
import Overview from './pages/dashboard/Overview.tsx';
import Consciousness from './pages/dashboard/Consciousness.tsx';
import Basins from './pages/dashboard/Basins.tsx';
import Graph from './pages/dashboard/Graph.tsx';
import Lifecycle from './pages/dashboard/Lifecycle.tsx';
import Cognition from './pages/dashboard/Cognition.tsx';
import Memory from './pages/dashboard/Memory.tsx';
import Telemetry from './pages/dashboard/Telemetry.tsx';
import Admin from './pages/dashboard/Admin.tsx';
import Training from './pages/dashboard/Training.tsx';
import Governor from './pages/dashboard/Governor.tsx';

const router = createBrowserRouter([
  {
    path: '/login',
    element: <Login />,
  },
  {
    element: <ProtectedRoute />,
    children: [
      {
        element: <Layout />,
        children: [
          { path: '/', element: <Navigate to="/chat" replace /> },
          { path: '/chat', element: <Chat /> },
          {
            path: '/dashboard',
            element: <Dashboard />,
            children: [
              { index: true, element: <Overview /> },
              { path: 'consciousness', element: <Consciousness /> },
              { path: 'basins', element: <Basins /> },
              { path: 'graph', element: <Graph /> },
              { path: 'lifecycle', element: <Lifecycle /> },
              { path: 'cognition', element: <Cognition /> },
              { path: 'memory', element: <Memory /> },
              { path: 'telemetry', element: <Telemetry /> },
              { path: 'training', element: <Training /> },
              { path: 'governor', element: <Governor /> },
              { path: 'admin', element: <Admin /> },
            ],
          },
          { path: '*', element: <Navigate to="/chat" replace /> },
        ],
      },
    ],
  },
]);

export default function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
}
