from typing import Literal
from pydantic import BaseModel, Field

class WorldAction(BaseModel):
    """An action and its arguments for the DiscoveryWorld agent (Use this when agent is not in dialogue)."""
    action: str = Field(description="Name of the action to perform, or null if ")
    arg1: int | str | None = Field(description="First argument, or null if unused")
    arg2: int | str | None = Field(description="Second argument, or null if unused")
    # chosen_dialog_option_int: int = Field(description="Number of the dialog option to select")
    
class DialogueChoice(BaseModel):
    """An option selected from the current dialog (Use this when agent is in dialogue)."""
    chosen_dialog_option_int: int = Field(description="Number of the dialog option to select")
    
class ContinueEpisode(BaseModel):
    """End this player's turn. The harness resumes it on the next wake-up."""

    turn_summary: str = Field(
        description="Briefly state what you did, observed, or learned this turn."
    )
    waiting_for: str | None = Field(
        default=None,
        description="A reply or event you expect, if any.",
    )

class FinishEpisode(BaseModel):
    """Request that the harness end the episode for all players."""

    reason: Literal["solved", "cannot_complete"]
    final_summary: str = Field(
        description="Give the findings and evidence that justify ending the episode."
    )