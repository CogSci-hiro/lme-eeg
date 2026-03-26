from __future__ import annotations


def format_convergence_rate(n_converged: int, n_features: int) -> str:
    """Format convergence counts for display."""
    if n_features == 0:
        return "0/0 (0.0%)"
    return f"{n_converged}/{n_features} ({100.0 * n_converged / n_features:.1f}%)"
