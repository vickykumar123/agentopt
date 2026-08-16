"""The improver: reflection-driven, evidence-gated agent improvement."""

from agentopt.improver.candidates import Candidate, CandidateError, EntryContent, overlay_for
from agentopt.improver.gate import GateResult, VerificationGate
from agentopt.improver.loop import ImproveLoop, ImproveOutcome, VerifiedChange
from agentopt.improver.reflector import Reflector

__all__ = [
    "Candidate",
    "CandidateError",
    "EntryContent",
    "GateResult",
    "ImproveLoop",
    "ImproveOutcome",
    "Reflector",
    "VerificationGate",
    "VerifiedChange",
    "overlay_for",
]
