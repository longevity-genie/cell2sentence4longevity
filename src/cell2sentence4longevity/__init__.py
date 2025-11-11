"""Cell to sentence preprocessing pipeline for longevity research."""

from cell2sentence4longevity.preprocess import app

__all__ = ["app"]


def main() -> None:
    """Entry point for the CLI."""
    app()
