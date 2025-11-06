"""
AeoLiS GUI Package - Modular GUI for AeoLiS Model

This package provides a modular graphical user interface for configuring
and visualizing AeoLiS aeolian sediment transport model results.

Modules:
- main: Main GUI application entry point
- config_manager: Configuration file I/O
- utils: Utility functions for file handling, time conversion, etc.
- visualizers: Visualization modules for different data types
"""

from aeolis.gui.main import launch_gui

__all__ = ['launch_gui']
