"""Feasibility-masked decoding, a separate baseline from the unmasked rows.

At every step only tokens that keep the answer a prefix of a feasible
``{"routes":[[...],...]}`` are allowed, the way the MTL policy decodes with the
environment's action mask. A customer can be written only if it is unvisited and fits
the current route: capacity, its time window, no linehaul after a backhaul, and (for a
closed route) a return to the depot within the distance limit and the depot's window.
A route can close once it has a customer; the answer can close once every customer is
visited. The arithmetic mirrors ``check_routes`` step by step, so a finished answer
passes it. This is a decoding constraint, not repair: nothing is changed after the fact.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

from vrp_service.env import EnvironmentInstance
from vrp_service.env.feasibility import TOLERANCE

HEADER = '{"routes":['
CHARS = frozenset(HEADER + "0123456789,]}")


@dataclass(frozen=True)
class _State:
    # header, list (expects "["), first (expects a customer), number, closed, end, done
    phase: str = "header"
    header: int = 0
    digits: str = ""
    visited: frozenset[int] = frozenset()
    # The route being written; position 0 is the depot.
    last: int = 0
    clock: float = 0.0
    length: float = 0.0
    linehaul: float = 0.0
    backhaul: float = 0.0
    seen_backhaul: bool = False


class RouteMask:
    """``vocab`` maps token id to its text; only tokens made of ``CHARS`` are ever allowed."""

    def __init__(self, instance: EnvironmentInstance, vocab: Iterable[tuple[int, str]], eos: int):
        self.instance = instance
        self.eos = eos
        self.by_first: dict[str, list[tuple[int, str]]] = {}
        for token, text in vocab:
            if text and set(text) <= CHARS:
                self.by_first.setdefault(text[0], []).append((token, text))
        self.position = {node: p for p, node in enumerate(instance.node_ids)}
        self.customers = instance.node_ids[1:]
        self._cache: dict[tuple, tuple[frozenset[str], frozenset[str]]] = {}

    def allowed(self, text: str) -> list[int]:
        """Tokens that may follow ``text``; only EOS once the answer is complete."""
        state = self._run(_State(), text)
        if state is None:
            return []
        if state.phase == "done":
            return [self.eos]
        return [
            token
            for char in self._next_chars(state)
            for token, piece in self.by_first.get(char, ())
            if self._run(state, piece) is not None
        ]

    def _next_chars(self, state: _State) -> str:
        if state.phase == "header":
            return HEADER[state.header]
        return {
            "list": "[",
            "first": "0123456789",
            "number": "0123456789,]",
            "closed": ",]",
            "end": "}",
        }[state.phase]

    def _run(self, state: _State | None, text: str) -> _State | None:
        for char in text:
            if state is None:
                return None
            state = self._step(state, char)
        return state

    def _step(self, s: _State, char: str) -> _State | None:
        if s.phase == "header":
            if char != HEADER[s.header]:
                return None
            done = s.header + 1 == len(HEADER)
            return replace(s, phase="list" if done else "header", header=s.header + 1)
        if s.phase == "list":
            fresh = replace(
                s, last=0, clock=0.0, length=0.0, linehaul=0.0, backhaul=0.0, seen_backhaul=False
            )
            return replace(fresh, phase="first") if char == "[" and self._ids(fresh)[0] else None
        if s.phase in ("first", "number") and char.isdigit():
            digits = s.digits + char
            return replace(s, phase="number", digits=digits) if digits in self._ids(s)[1] else None
        if s.phase == "number" and char in ",]":
            if s.digits not in self._ids(s)[0]:
                return None
            s = self._visit(s, int(s.digits))
            if char == "]":
                return replace(s, phase="closed")
            return replace(s, phase="first") if self._ids(s)[0] else None
        if s.phase == "closed":
            if char == ",":
                return replace(s, phase="list") if len(s.visited) < len(self.customers) else None
            if char == "]":
                return replace(s, phase="end") if len(s.visited) == len(self.customers) else None
        if s.phase == "end" and char == "}":
            return replace(s, phase="done")
        return None

    def _ids(self, s: _State) -> tuple[frozenset[str], frozenset[str]]:
        """Customers that may come next, as strings, and every prefix of them."""
        key = (s.visited, s.last, s.clock, s.length, s.linehaul, s.backhaul, s.seen_backhaul)
        if key not in self._cache:
            fits = [n for n in self.customers if n not in s.visited and self._fits(s, n)]
            ids = frozenset(map(str, fits))
            prefixes = frozenset(i[:k] for i in ids for k in range(1, len(i) + 1))
            self._cache[key] = ids, prefixes
        return self._cache[key]

    def _visit(self, s: _State, node: int) -> _State:
        inst, p = self.instance, self.position[node]
        leg = math.dist(inst.locs[s.last], inst.locs[p])
        clock = max(s.clock + leg / inst.speed, inst.time_windows[p][0]) + inst.service_time[p]
        return replace(
            s,
            digits="",
            visited=s.visited | {node},
            last=p,
            clock=clock,
            length=s.length + leg,
            linehaul=s.linehaul + inst.demand_linehaul[p],
            backhaul=s.backhaul + inst.demand_backhaul[p],
            seen_backhaul=s.seen_backhaul or inst.demand_backhaul[p] > 0,
        )

    def _fits(self, s: _State, node: int) -> bool:
        """The same checks, in the same float order, as ``check_routes``."""
        inst, p = self.instance, self.position[node]
        capacity = inst.vehicle_capacity * (1 + TOLERANCE)
        if inst.demand_backhaul[p] == 0 and s.seen_backhaul and inst.demand_linehaul[p] > 0:
            return False
        leg = math.dist(inst.locs[s.last], inst.locs[p])
        if s.clock + leg / inst.speed > inst.time_windows[p][1] + TOLERANCE:
            return False
        after = self._visit(s, node)
        if after.linehaul > capacity or after.backhaul > capacity:
            return False
        if inst.open_route:
            return after.length <= inst.distance_limit + TOLERANCE
        back = math.dist(inst.locs[p], inst.locs[0])
        return (
            after.length + back <= inst.distance_limit + TOLERANCE
            and after.clock + back / inst.speed <= inst.time_windows[0][1] + TOLERANCE
        )
