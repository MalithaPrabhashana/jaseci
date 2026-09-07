"""The `conversation=<list>` kwarg: what byLLM writes back into the caller's list.

The caller passes a list and byLLM appends to it in place, so these tests assert
on that list directly. Scaffolding (the system prompt) and byLLM's own finish_tool
plumbing are filtered out on the way; a real tool result is not.
"""

import from support_tests { FakeLLM, finish, call, say }


def get_weather(city: str) -> str {
    return f"sunny in {city}";
}


"""Every message in `history` with the given role."""
def by_role(history: list, role: str) -> list {
    return [
        m
        for m in history
        if m.get("role") == role
    ];
}


"""One conversation carried across two turns, sharing the caller's list."""
def ask_twice(history: list) -> tuple {
    llm = FakeLLM(
        replies=[
            call("get_weather", {"city": "Paris"}),
            finish("Paris is sunny."),
            finish("Yes, still sunny in Paris.")
        ]
    );

    def ask(message: str) -> str by llm(tools=[get_weather], conversation=history);

    first = ask("How is Paris?");
    turn1_len = len(history);
    second = ask("And again?");
    return (llm, first, second, turn1_len);
}


test "the caller's list is mutated in place, not replaced" {
    history: list = [{"role": "user", "content": "PRIOR_SESSION_USER_TURN"}];
    before = id(history);

    ask_twice(history);

    assert id(history) == before , "write_back_conversation must mutate the caller's list";
    assert any(
        "PRIOR_SESSION_USER_TURN" in str(m.get("content", ""))
        for m in by_role(history, "user")
    ) , f"a pre-seeded dict must survive the round trip: {history}";
}


test "scaffolding and finish_tool are filtered out, a real tool result is not" {
    history: list = [{"role": "user", "content": "PRIOR_SESSION_USER_TURN"}];
    ask_twice(history);

    roles = [m.get("role") for m in history];
    assert "system" not in roles , f"the system prompt is scaffolding, not conversation: {roles}";

    tool_msgs = by_role(history, "tool");
    assert len(tool_msgs) == 1 , f"expected the one real tool result, got {len(
        tool_msgs
    )}";
    assert tool_msgs[0].get("name") == "get_weather" , (
        f"only finish_tool should be filtered, got name={tool_msgs[0].get('name')!r}"
    );
    assert "tool_call_id" in tool_msgs[0] , (
        f"a tool result needs its call id to be replayable: {tool_msgs[0]}"
    );
    for m in history {
        assert m.get("name") != "finish_tool" , f"finish_tool is plumbing and leaked: {m}";
    }
}


test "a second turn grows the same list and keeps every earlier message" {
    history: list = [{"role": "user", "content": "PRIOR_SESSION_USER_TURN"}];
    (llm, first, second, turn1_len) = ask_twice(history);

    assert first == "Paris is sunny." , f"got {first!r}";
    assert second == "Yes, still sunny in Paris." , f"got {second!r}";
    assert len(history) > turn1_len , "the second turn must append, not overwrite";

    users = by_role(history, "user");
    assert len(users) == 3 , (
        f"expected the pre-seeded turn plus two asked turns, got {len(users)}"
    );
    contents = [str(m.get("content", "")) for m in users];
    assert any("PRIOR_SESSION_USER_TURN" in c for c in contents) , (
        f"the pre-seeded message was dropped on turn 2: {contents}"
    );
    assert any("Paris" in c for c in contents) , f"turn 1's message was dropped: {contents}";
}
