import { useEffect } from 'react';

export function Toast({ message, onDismiss, duration = 3000 }) {
  useEffect(() => {
    if (!message) return;
    const timer = setTimeout(() => {
      onDismiss();
    }, duration);
    return () => clearTimeout(timer);
  }, [message, onDismiss, duration]);

  if (!message) return null;

  return (
    <div
      className="toast-notification"
      role="status"
      aria-live="polite"
      onClick={onDismiss}
    >
      <span className="toast-text">{message}</span>
    </div>
  );
}
