"""
Base command handler for Interactive Knowledge Transfer.
"""

import re
from typing import List, Optional
from ..core.data_types import Colors, SessionState


class BaseCommandHandler:
    """Base class for command handlers."""

    def __init__(self, model, tokenizer, device, session: SessionState):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.session = session

    def parse_command(self, command: str) -> List[str]:
        """Parse command into parts, handling quoted strings."""
        parts = []
        pattern = r'"([^"]*)"|([^\s"]+)'
        matches = re.findall(pattern, command)

        for match in matches:
            # match is a tuple (quoted_content, unquoted_content)
            if match[0]:  # If quoted content exists
                parts.append(match[0])
            elif match[1]:  # If unquoted content exists
                parts.append(match[1])

        return parts

    def error(self, message: str):
        """Display error message."""
        print(f"{Colors.RED}{message}{Colors.END}")

    def success(self, message: str):
        """Display success message."""
        print(f"✓ {message}")

    def warning(self, message: str):
        """Display warning message."""
        print(f"{Colors.YELLOW}{message}{Colors.END}")

    def info(self, message: str):
        """Display info message."""
        print(f"{Colors.CYAN}{message}{Colors.END}")