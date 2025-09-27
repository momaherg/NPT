"""
Token tracking commands for Interactive Knowledge Transfer.
"""

from typing import List
from .base import BaseCommandHandler
from ..core.data_types import Colors


class TrackingCommands(BaseCommandHandler):
    """Handles token tracking commands."""

    def cmd_track(self, args: List[str]):
        """Track token probabilities."""
        if not args:
            # Show currently tracked tokens
            if self.session.tracked_tokens:
                print(f"Tracked tokens: {Colors.CYAN}{', '.join(repr(t) for t in self.session.tracked_tokens)}{Colors.END}")
            else:
                print("No tokens being tracked")
        else:
            # Add tokens to tracking
            added_tokens = []
            for token in args:
                if token not in self.session.tracked_tokens:
                    self.session.tracked_tokens.append(token)
                    # Tokenize to get ID - handle tokens with/without spaces
                    token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
                    if len(token_ids) > 0:
                        # Use the first token ID if multiple
                        self.session.tracked_token_ids.append(token_ids[0])
                        added_tokens.append(token)
                    else:
                        # Failed to tokenize, remove from tracking
                        self.session.tracked_tokens.pop()
                        self.warning(f"Could not tokenize '{token}'")
                else:
                    self.warning(f"Already tracking: {token}")

            if added_tokens:
                self.success(f"Tracking tokens: {Colors.CYAN}{', '.join(repr(t) for t in added_tokens)}{Colors.END}")

    def cmd_untrack(self, args: List[str]):
        """Stop tracking specific tokens."""
        if not args:
            self.error("Usage: untrack <token>")
            return

        for token in args:
            if token in self.session.tracked_tokens:
                idx = self.session.tracked_tokens.index(token)
                self.session.tracked_tokens.pop(idx)
                self.session.tracked_token_ids.pop(idx)
                self.success(f"Stopped tracking: {Colors.CYAN}{token}{Colors.END}")
            else:
                self.warning(f"Token not tracked: {token}")

    def cmd_clear_tracking(self):
        """Clear all tracked tokens."""
        self.session.tracked_tokens.clear()
        self.session.tracked_token_ids.clear()
        self.success("Cleared all tracked tokens")