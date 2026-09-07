"""`visit ... by llm()`: what the routing prompt offers, and what gets visited.

The model here reads the prompt and chooses from it, rather than being handed a
scripted answer. Handles are generated from the candidate set, so scripting them
is brittle, and reading them proves the prompt carries what a chooser needs.
"""

import os;
import from jaclang { JacRuntimeInterface as Jac }
import from jaclang.runtime.runtime { JacRuntime }
import from fixtures.routing_graph {
    Agent,
    Hub,
    Route,
    collector,
    leaky,
    picker,
    router
}
import fixtures.routing_graph as rg;
import from support_tests { RoutingLLM, routing_candidates }


"""A hub with one agent per (tag, edge priority) pair."""
def build_hub(pairs: list) -> Hub {
    hub = root ++> Hub();
    for (tag, priority) in pairs {
        hub +>:Route(priority=priority):+> Agent(tag=tag);
    }
    return hub;
}


test "the prompt renders each candidate's edge, so routing can choose by it" {
    llm = RoutingLLM(
        pick=lambda (c: list) { [
            x[0]
            for x in c
            if "priority='high'" in x[1]
        ]; }
    );
    rg.llm = llm;
    hub = build_hub([("alpha", "low"), ("beta", "high")]);

    w = router() spawn hub;

    assert w.visited == ["beta"] , f"the high-priority edge should win, got {w.visited}";
    cands = routing_candidates(llm.seen[0]);
    assert len(cands) == 2 , f"both successors should be offered, got {cands}";
    assert any("priority='high'" in line for (_h, line) in cands) , (
        f"the edge attribute must be rendered, got {cands}"
    );
}


test "an arbitrary node list is offered without a fabricated edge" {
    # `visit` accepts any node list, not just [-->] successors. Standalone nodes
    # have no edge to `here`, so the prompt must not invent one.
    llm = RoutingLLM();
    rg.llm = llm;
    hub = root ++> Hub();
    loose = [Agent(tag="alpha"), Agent(tag="beta")];

    w = picker(options=loose) spawn hub;

    assert len(w.visited) == 1 , f"select=1 should visit one, got {w.visited}";
    lines = " ".join([line for (_h, line) in routing_candidates(llm.seen[0])]);
    for invented in ["(direct)", "here --", "here <-"] {
        assert invented not in lines , (
            f"a node-only candidate must not be framed as an edge: {lines!r}"
        );
    }
}


test "candidates with identical field values stay distinct targets" {
    # Two parameterless Agents are equal by value. Value-based dedup would
    # collapse them and visit once; identity-based dedup keeps both.
    llm = RoutingLLM(pick=lambda (c: list) { [x[0] for x in c]; });
    rg.llm = llm;
    hub = build_hub([("same", "low"), ("same", "low")]);

    w = collector() spawn hub;

    assert len(w.visited) == 2 , (
        f"two equal-valued nodes are two targets, got {w.visited}"
    );
}


test "routing-only call params are stripped before the request goes out" {
    # select, intent and incl_info are consumed by the router. A real backend
    # rejects them as unknown kwargs. temperature is a real knob and must survive.
    llm = RoutingLLM();
    rg.llm = llm;
    hub = build_hub([("alpha", "low")]);

    leaky() spawn hub;

    sent = llm.seen[0];
    for leaked in ["select", "intent", "incl_info"] {
        assert leaked not in sent , f"{leaked} must not reach the model: {sorted(
            sent
        )}";
    }
    assert sent.get("temperature") == 0.5 , (
        f"a real API knob must still pass through, got {sent.get('temperature')!r}"
    );
}


test "a routing call dispatches through the central call_llm entry" {
    # route_visit must go through Jac.call_llm, not model.invoke, so the one
    # runtime entry point stays authoritative for routing as it is for `by llm`.
    llm = RoutingLLM();
    rg.llm = llm;
    hub = build_hub([("a", "low")]);

    calls: list = [];
    original = JacRuntime.call_llm;

    def counting(model: any, mt_run: any) -> any {
        calls.append(1);
        return original(model, mt_run);
    }
    JacRuntime.call_llm = counting;
    try {
        w = router() spawn hub;
    } finally {
        JacRuntime.call_llm = original;
    }
    assert len(calls) == 1 , f"expected exactly one call_llm dispatch, got {len(
        calls
    )}";
    assert w.visited == ["a"] , f"got {w.visited}";
}
