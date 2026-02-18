import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
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

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<ProtectedRoute />}>
            <Route element={<Layout />}>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/dashboard" element={<Dashboard />}>
                <Route index element={<Overview />} />
                <Route path="consciousness" element={<Consciousness />} />
                <Route path="basins" element={<Basins />} />
                <Route path="graph" element={<Graph />} />
                <Route path="lifecycle" element={<Lifecycle />} />
                <Route path="cognition" element={<Cognition />} />
                <Route path="memory" element={<Memory />} />
                <Route path="telemetry" element={<Telemetry />} />
                <Route path="training" element={<Training />} />
                <Route path="admin" element={<Admin />} />
              </Route>
              <Route path="*" element={<Navigate to="/chat" replace />} />
            </Route>
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
