"""
Interactive Knowledge Transfer Tool for NPT Models.

A modular REPL interface for extracting, storing, and injecting
modulations to explore knowledge transfer in NPT models.
"""

from .cli import InteractiveKnowledgeTransfer, main

__all__ = ['InteractiveKnowledgeTransfer', 'main']