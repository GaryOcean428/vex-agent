import { Outlet } from 'react-router-dom';

export default function Dashboard() {
  return (
    <div className="dashboard-page">
      <Outlet />
    </div>
  );
}
