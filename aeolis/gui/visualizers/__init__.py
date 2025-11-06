"""
Visualizers package for AeoLiS GUI.

This package contains specialized visualizer modules for different types of data:
- domain: Domain setup visualization (bed, vegetation, etc.)
- wind: Wind input visualization (time series, wind roses)
- output_2d: 2D output visualization
- output_1d: 1D transect visualization
"""

from aeolis.gui.visualizers.domain import DomainVisualizer

__all__ = ['DomainVisualizer']
