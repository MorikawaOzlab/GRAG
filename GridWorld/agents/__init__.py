"""Expose the shared, user-maintained agents from the archived application."""
from pathlib import Path
__path__ = [str(Path(__file__).resolve().parents[2] / 'agents')]
