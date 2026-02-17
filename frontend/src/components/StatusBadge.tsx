interface StatusBadgeProps {
  label: string;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

const VARIANT_CLASSES: Record<string, string> = {
  default: 'badge-default',
  success: 'badge-success',
  warning: 'badge-warning',
  error: 'badge-error',
  info: 'badge-info',
};

export default function StatusBadge({ label, variant = 'default' }: StatusBadgeProps) {
  return (
    <span className={`status-badge ${VARIANT_CLASSES[variant]}`}>
      {label}
    </span>
  );
}
