"""The agentic communication network (README §"Agentic communication networks")
as a typed, logged message bus.

The README draws forward/backward edges between modules in prose; this makes
them concrete and enforces two things the prose only implied:

  * **Direction is explicit.** A `Message` is FORWARD (knowledge/ideas pushed
    downstream to engineers) or BACKWARD (requests/findings pushed upstream to
    scientists). This lets us honor — and audit — the README's directionality
    rules, and resolve its one contradiction (critique point 12): the M7->M2
    backward edge for mechanistic insight that reshapes hypotheses IS allowed and
    is declared here, rather than being both asserted and denied.

  * **Edges are declared.** `ALLOWED` is the adjacency the README specifies.
    `send` refuses an edge that is not declared, so the network topology is a
    checked contract, not an emergent free-for-all.

Transport is intentionally trivial (in-memory, append-only log). The point is the
contract and the audit trail, not the plumbing — swap in Academy/asyncio queues
behind `Bus` without touching callers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger("asop.bus")

Direction = Literal["forward", "backward"]

# Declared edges (src, dst). Forward = toward engineers; backward = toward
# scientists/librarian. Mirrors the README, with M7->M2 backward kept.
ALLOWED: set[tuple[str, str, Direction]] = {
    # M1 librarian <-> everyone (forward: supply knowledge; backward: requests)
    *[("M1", m, "forward") for m in ("M2", "M3", "M4", "M5", "M6", "M7")],
    *[(m, "M1", "backward") for m in ("M2", "M3", "M4", "M5", "M6", "M7")],
    # M2 hypothesis <-> engineers (forward: ideas; backward: was it useful?)
    *[("M2", m, "forward") for m in ("M3", "M4", "M5", "M6", "M7")],
    *[(m, "M2", "backward") for m in ("M3", "M4", "M5", "M6")],
    # M7 mechanism -> engineers (forward: which binders are mechanistically good)
    *[("M7", m, "forward") for m in ("M3", "M4", "M5", "M6")],
    # M7 -> M2 backward: mechanistic insight reshapes hypotheses (critique 12)
    ("M7", "M2", "backward"),
}


@dataclass
class Message:
    src: str
    dst: str
    direction: Direction
    topic: str
    payload: Any = None
    ts: float = field(default_factory=time.time)


@dataclass
class Bus:
    """In-memory, append-only, topology-checked message bus."""

    allowed: set[tuple[str, str, Direction]] = field(default_factory=lambda: set(ALLOWED))
    log_: list[Message] = field(default_factory=list)
    _inbox: dict[str, list[Message]] = field(default_factory=dict)

    def send(self, src: str, dst: str, direction: Direction, topic: str, payload: Any = None) -> Message:
        if (src, dst, direction) not in self.allowed:
            raise PermissionError(
                f"undeclared edge {src} -{direction}-> {dst} (topic {topic!r}); "
                f"add it to ALLOWED if the network should carry it"
            )
        msg = Message(src=src, dst=dst, direction=direction, topic=topic, payload=payload)
        self.log_.append(msg)
        self._inbox.setdefault(dst, []).append(msg)
        log.info("%s -%s-> %s : %s", src, direction, dst, topic)
        return msg

    def inbox(self, module: str, *, direction: Direction | None = None,
              drain: bool = False) -> list[Message]:
        msgs = list(self._inbox.get(module, []))
        if direction is not None:
            msgs = [m for m in msgs if m.direction == direction]
        if drain:
            keep = [m for m in self._inbox.get(module, []) if m not in msgs]
            self._inbox[module] = keep
        return msgs

    def history(self, src: str | None = None, dst: str | None = None) -> list[Message]:
        return [m for m in self.log_
                if (src is None or m.src == src) and (dst is None or m.dst == dst)]
