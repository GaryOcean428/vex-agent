import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

/* ─── Types ─── */

interface ErrorBoundaryProps {
  children: ReactNode;
  /** Optional custom fallback renderer. Receives the error and a reset callback. */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/* ─── Full-page ErrorBoundary ─── */

/**
 * Catches render errors in the subtree and shows a full-page fallback.
 * Use this at the app or route level.
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("[ErrorBoundary] Uncaught error:", error, info.componentStack);
  }

  private handleReset = (): void => {
    this.setState({ error: null });
  };

  private handleGoHome = (): void => {
    this.setState({ error: null });
    window.location.href = "/chat";
  };

  render(): ReactNode {
    const { error } = this.state;
    const { children, fallback } = this.props;

    if (error === null) {
      return children;
    }

    if (fallback) {
      return fallback(error, this.handleReset);
    }

    return (
      <div className="error-boundary">
        <div className="error-boundary-card">
          <div className="error-boundary-icon" aria-hidden="true">
            !
          </div>
          <h1 className="error-boundary-title">Something went wrong</h1>
          <p className="error-boundary-message">
            An unexpected error occurred. You can try again or return to the
            chat.
          </p>
          {error.message && (
            <div className="error-boundary-details">
              <code>{error.message}</code>
            </div>
          )}
          <div className="error-boundary-actions">
            <button
              type="button"
              className="error-boundary-btn error-boundary-btn-primary"
              onClick={this.handleReset}
            >
              Try Again
            </button>
            <button
              type="button"
              className="error-boundary-btn error-boundary-btn-secondary"
              onClick={this.handleGoHome}
            >
              Go Home
            </button>
          </div>
        </div>
      </div>
    );
  }
}

/* ─── Inline ComponentErrorBoundary ─── */

interface ComponentErrorBoundaryProps {
  children: ReactNode;
  /** Label shown in the inline error. Defaults to "Component error". */
  name?: string;
}

interface ComponentErrorBoundaryState {
  error: Error | null;
}

/**
 * Lightweight error boundary for wrapping individual components.
 * Shows a compact inline error instead of a full-page fallback.
 */
class ComponentErrorBoundary extends Component<
  ComponentErrorBoundaryProps,
  ComponentErrorBoundaryState
> {
  constructor(props: ComponentErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ComponentErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(
      `[ComponentErrorBoundary${this.props.name ? `: ${this.props.name}` : ""}]`,
      error,
      info.componentStack,
    );
  }

  private handleRetry = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    const { children, name } = this.props;

    if (error === null) {
      return children;
    }

    return (
      <div className="component-error" role="alert">
        <span className="component-error-icon" aria-hidden="true">
          !
        </span>
        <div className="component-error-text">
          <div className="component-error-label">
            {name ?? "Component error"}
          </div>
          {error.message && (
            <div className="component-error-message">{error.message}</div>
          )}
        </div>
        <button
          type="button"
          className="component-error-retry"
          onClick={this.handleRetry}
        >
          Retry
        </button>
      </div>
    );
  }
}

export { ErrorBoundary, ComponentErrorBoundary };
