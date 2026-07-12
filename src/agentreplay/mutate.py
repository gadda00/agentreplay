"""Counterfactual mutation engine.

Takes a recorded cassette, applies a *patch* (a replacement response for
one or more steps), and produces a new cassette that can be replayed to
see how the rest of the trajectory changes.

This is the "edit one step, replay forward" capability from §5.4 of the
product proposal — the direct answer to incident-review questions like
"would the agent still have taken the harmful action if the tool had
returned a permission-denied error instead of success?".

Mutation works in two flavours:

    * :func:`mutate_response` — replace the recorded response at one
      step. The request hash is preserved, so the call-site ID stays
      matchable and the upstream trajectory replays bit-exact.

    * :func:`mutate_and_replay` — apply a mutation, then run the agent
      forward in HYBRID mode to see where the trajectory diverges from
      the original recording. Everything up to the mutated step is free;
      everything after is either served from the cassette (if the
      request still matches) or falls through to a live call.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from agentreplay.cassette import Cassette
from agentreplay.constants import Mode
from agentreplay.errors import MutationError
from agentreplay.replayer import Replayer


def mutate_response(
    source: Union[str, os.PathLike, Cassette],
    *,
    seq: Optional[int] = None,
    step_id: Optional[str] = None,
    call_id: Optional[str] = None,
    new_response: Any,
    target_root: Optional[Union[str, os.PathLike]] = None,
    new_id: Optional[str] = None,
) -> Cassette:
    """Create a forked cassette with the response at the targeted step replaced.

    Exactly one of ``seq`` / ``step_id`` / ``call_id`` must be provided.

    Returns the new (writable) :class:`Cassette`. The caller can then
    open it with a :class:`Replayer` in HYBRID mode to run the agent
    forward from the mutation point.
    """
    src = source if isinstance(source, Cassette) else Cassette.open(source, readonly=True)

    # Resolve the target step.
    if seq is None:
        if call_id is not None:
            ev = src.events.by_call_id(call_id)
            if ev is None:
                raise MutationError(f"no event with call_id={call_id!r} in cassette {src.meta.id}")
            seq = ev.seq
        elif step_id is not None:
            matches = src.events.by_step(step_id)
            if not matches:
                raise MutationError(f"no event with step_id={step_id!r} in cassette {src.meta.id}")
            if len(matches) > 1:
                raise MutationError(
                    f"step_id {step_id!r} matches {len(matches)} events; pass seq= to disambiguate"
                )
            seq = matches[0].seq
        else:
            raise MutationError("one of seq / step_id / call_id must be provided")

    # Decide where to write the fork.
    if target_root is None:
        target_root = Path(src.root).with_suffix(".mut")
    target_root = Path(target_root)
    if target_root.exists() and any(target_root.iterdir()):
        # Pick a fresh sibling path.
        target_root = target_root.with_name(target_root.name + f".{new_id or 'mut'}")

    forked = src.fork(target_root, new_id=new_id)
    forked.replace_response(seq, new_response)
    forked.meta.tags = list(set(forked.meta.tags + ["mutated"]))
    forked.meta.extra = {**forked.meta.extra, "mutated_from": src.meta.id, "mutated_seq": seq}
    forked.save()
    return forked


def mutate_and_replay(
    source: Union[str, os.PathLike, Cassette],
    *,
    agent_run: Callable[[Replayer], Any],
    new_response: Any,
    seq: Optional[int] = None,
    step_id: Optional[str] = None,
    call_id: Optional[str] = None,
    target_root: Optional[Union[str, os.PathLike]] = None,
    live_client: Any = None,
    live_http: Any = None,
) -> Dict[str, Any]:
    """Apply a mutation, then run the agent forward in HYBRID mode.

    The ``agent_run`` callable receives a :class:`Replayer` configured
    in HYBRID mode against the mutated cassette. It should run the
    agent's code, using the replayer's ``wrap_*`` methods to plug in
    the recording proxies. The returned dict contains:

        - ``cassette``: the mutated (forked) cassette
        - ``result``  : whatever ``agent_run`` returned
        - ``divergences``: list of divergence points hit during replay
    """
    mutated = mutate_response(
        source,
        seq=seq,
        step_id=step_id,
        call_id=call_id,
        new_response=new_response,
        target_root=target_root,
    )
    replayer = Replayer.open(
        mutated.root,
        mode=Mode.HYBRID,
        live_client=live_client,
        live_http=live_http,
    )
    result = agent_run(replayer)
    return {
        "cassette": mutated,
        "result": result,
        "divergences": replayer.divergences,
    }


def apply_patch_set(
    source: Union[str, os.PathLike, Cassette],
    patches: List[Dict[str, Any]],
    *,
    target_root: Optional[Union[str, os.PathLike]] = None,
) -> Cassette:
    """Apply multiple mutations in one shot.

    ``patches`` is a list of dicts, each with the same keys as
    :func:`mutate_response` (``seq`` / ``step_id`` / ``call_id`` plus
    ``new_response``). The patches are applied to a single fork in
    ascending ``seq`` order so they do not interfere with each other.
    """
    src = source if isinstance(source, Cassette) else Cassette.open(source, readonly=True)
    if target_root is None:
        target_root = Path(src.root).with_suffix(".patched")
    forked = src.fork(target_root)
    # Sort patches by seq so we apply them in order.
    resolved: list[tuple[int, Any]] = []
    for p in patches:
        seq = p.get("seq")
        if seq is None:
            ev = None
            if p.get("call_id"):
                ev = src.events.by_call_id(p["call_id"])
            elif p.get("step_id"):
                ms = src.events.by_step(p["step_id"])
                if len(ms) == 1:
                    ev = ms[0]
            if ev is None:
                raise MutationError(f"patch {p!r} does not resolve to a unique event")
            seq = ev.seq
        resolved.append((int(seq), p["new_response"]))
    for seq, _ in resolved:
        pass  # validation only
    for seq, new_response in sorted(resolved, key=lambda x: x[0]):
        forked.replace_response(seq, new_response)
    forked.meta.tags = list(set(forked.meta.tags + ["patched"]))
    forked.meta.extra = {**forked.meta.extra, "patched_from": src.meta.id, "num_patches": len(patches)}
    forked.save()
    return forked
