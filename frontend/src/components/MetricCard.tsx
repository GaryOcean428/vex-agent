import type { CSSProperties } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  color?: string;
  subtitle?: string;
  progress?: number; // 0..1
  threshold?: string;
}

export default function MetricCard({ label, value, color, subtitle, progress, threshold }: MetricCardProps) {
  return (
    <div className="metric-card" style={{ '--card-color': color ?? 'var(--accent)' } as CSSProperties}>
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value" style={{ color: color ?? 'var(--text)' }}>
        {typeof value === 'number' ? value.toFixed(3) : value}
      </div>
      {progress !== undefined && (
        <div className="metric-card-bar">
          <div
            className="metric-card-bar-fill"
            style={{ width: `${Math.min(progress * 100, 100)}%`, background: color ?? 'var(--accent)' }}
          />
        </div>
      )}
      {threshold && <div className="metric-card-threshold">{threshold}</div>}
      {subtitle && <div className="metric-card-subtitle">{subtitle}</div>}
    </div>
  );
}
