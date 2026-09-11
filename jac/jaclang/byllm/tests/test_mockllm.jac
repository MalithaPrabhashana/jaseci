"""MockLLM's own contract: what each queued output becomes, and what is recorded.

MockLLM is public API that users test their own apps with, so each promise in the
byLLM reference's "What each output becomes" table is pinned here.
"""

import from jaclang.byllm.lib {
    MockLLM,
    Model,
    MockToolCall,
    MockRawResponse,
    MockError,
    StreamEvent
}
import from jaclang.byllm.exceptions { OutputConversionError }


obj Person {
    has name: str,
        age: int;
}

enum Mood { HAPPY = "happy", SAD = "sad" }


test "a string answers a str call, and every request is recorded" {
    # outputs= and the older config={"outputs": ...} build the same queue.
    for m in [
        MockLLM(outputs=["Hello Ada"]),
        MockLLM(model_name="mockllm", config={"outputs": ["Hello Ada"]})
    ] {
        def greet(name: str) -> str by m();

        assert greet("Ada") == "Hello Ada";
        assert len(m.seen) == 1;
        assert "Ada" in str(m.sent("messages")[0]) , "the request must be the one byLLM built";
        assert "Ada" in m.seen_prompts[0];
        assert m.exhausted();
    }
}


test "typed values round-trip through the real parser" {
    m = MockLLM(outputs=[Person(name="Ada", age=36), 36, Mood.SAD, [1, 2]]);
    def who -> Person by m();
    def age -> int by m();
    def mood -> Mood by m();
    def pair -> list[int] by m();

    p = who();
    assert isinstance(p, Person) and p.name == "Ada" and p.age == 36 , f"got {p!r}";
    assert m.sent("response_format")[0] is not None , "a typed call must request a schema";
    assert age() == 36;
    assert mood() == Mood.SAD , "an enum travels by value and comes back as the member";
    assert pair() == [1, 2];
}


test "a bare string for a typed return is parsed like model text, not trusted" {
    m = MockLLM(outputs=["36"]);
    def age -> int by m();
    raised = False;
    try {
        age();
    } except (OutputConversionError, IndexError) {
        raised = True;
    }
    assert raised , "'36' is not the envelope a real model sends for an int";
}


test "a tool call resolves by name against the tools the call offers" {
    for (label, queued, expect_ran) in [
        ("offered, by function", "add", True),
        ("offered, by name", "add-by-name", True),
        ("not offered", "secret", False)
    ] {
        ran: list = [];
        def add(a: int, b: int) -> int {
            ran.append("add");
            return a + b;
        }
        def secret -> str {
            ran.append("secret");
            return "leaked";
        }
        tool = add
            if queued == "add"
            else ("add" if queued == "add-by-name" else secret);
        args = {} if queued == "secret" else {"a": 2, "b": 3};
        m = MockLLM(
            outputs=[
                MockToolCall(tool=tool, args=args),
                MockToolCall(tool="finish_tool", args={"final_output": "done"})
            ]
        );
        def calc(q: str) -> str by m(tools=[add]);

        assert calc("2+3") == "done" , label;
        assert (len(ran) > 0) == expect_ran , f"{label}: ran {ran}";
        if not expect_ran {
            assert "unknown tool" in str(m.sent("messages")[1]) , (
                f"{label}: the model must be told the tool is unknown"
            );
        }
    }
}


test "a streamed answer arrives in chunks, with queued usage attached" {
    m = MockLLM(
        outputs=[
            (
                "Paris is sunny right now.",
                {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            )
        ]
    );
    def ask(q: str) -> str by m(stream=True, logging=True);

    events = list(ask("weather"));
    chunks = [
        e
        for e in events
        if isinstance(e, StreamEvent) and e.event_type == "chunk"
    ];
    usage = [
        e
        for e in events
        if isinstance(e, StreamEvent) and e.event_type == "usage"
    ];
    assert len(chunks) > 1 , f"expected several chunks, got {len(chunks)}";
    assert "".join([str(c.data.get("content", "")) for c in chunks])
        == "Paris is sunny right now.";
    assert usage and usage[-1].data["total"]["total_tokens"] == 10 , f"got {usage}";
}


test "a queued error is raised, and an empty queue says which call ran out" {
    for (label, outputs, kind, fragment) in [
        ("queued error", [MockError(error=RuntimeError("boom"))], RuntimeError, "boom"),
        ("empty queue", [], IndexError, "no queued output left for call 1")
    ] {
        m = MockLLM(outputs=outputs);
        def ask(q: str) -> str by m();
        message = "";
        try {
            ask("x");
        } except kind as e {
            message = str(e);
        }
        assert fragment in message , f"{label}: got {message!r}";
    }
}


test "MockRawResponse sends model text verbatim into the parser" {
    m = MockLLM(
        outputs=[
            MockRawResponse(
                content='{"schema_object_wrapper": {"name": "Bo", "age": 4}}'
            )
        ]
    );
    def who -> Person by m();
    p = who();
    assert p.name == "Bo" and p.age == 4;
}


test "Model(model_name='mockllm') delegates to MockLLM" {
    m = Model(model_name="mockllm", config={"outputs": ["via model"]});
    def ask(q: str) -> str by m();
    assert ask("x") == "via model";
}
