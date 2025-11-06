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

# Import from the parent-level gui.py module to avoid naming conflicts
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the legacy AeolisGUI class and related variables from gui.py at aeolis level
import importlib.util
gui_py_path = os.path.join(parent_dir, 'gui.py')
spec = importlib.util.spec_from_file_location("aeolis_gui_module", gui_py_path)
gui_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui_module)

AeolisGUI = gui_module.AeolisGUI
configfile = gui_module.configfile
dic = gui_module.dic

__all__ = ['launch_gui', 'AeolisGUI', 'configfile', 'dic']
