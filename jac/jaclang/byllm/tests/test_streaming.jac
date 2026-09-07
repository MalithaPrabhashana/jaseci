"""Streaming: what a `by llm(stream=True)` call hands back.

Without `logging=True` the call yields text. With it, the call yields StreamEvents
and the answer arrives on the `chunk` channel; tool activity arrives before it.
"""

import asyncio;
import contextvars;
import from jaclang.byllm.exceptions { ConfigurationError }
import from jaclang.byllm.telemetry { _current_invocation_id }
import from support_tests {
    FakeLLM,
    call,
    chunk_text,
    event_types,
    events_of,
    finish,
    say,
    stream_events,
    typed
}


test "a streaming call yields the answer in pieces, not all at once" {
    essay = (
        "The orca whale, or killer whale, is one of the most intelligent and "
        "adaptable marine predators in the ocean. Renowned for their complex "
        "social structures and sophisticated hunting techniques, orcas exhibit "
        "remarkable teamwork while pursuing prey."
    );
    llm = FakeLLM(replies=[say(essay)], chunk_size=12);

    """Generate a short essay about the given topic."""
    def generate_essay(topic: str) -> str by llm(stream=True);

    stream = generate_essay("The orca whale and its hunting techniques");
    assert not isinstance(stream, str) , (
        "a streaming call must hand back an iterator, not the finished string"
    );

    chunks = list(stream);
    assert len(chunks) > 1 , (
        f"the answer should arrive in pieces, got {len(chunks)}: nothing streamed"
    );
    assert "".join(chunks) == essay , (
        f"the pieces must reassemble into the answer, got {''.join(chunks)[:80]!r}"
    );
}


def get_weather(city: str) -> str {
    return f"The weather in {city} is sunny, 25 degrees.";
}


test "a logging stream with no tools yields chunks then one usage event" {
    llm = FakeLLM(
        replies=[say("Jac is a programming language designed for the AI era.")]
    );

    """Explain what Jac is in one sentence."""
    def explain_jac(topic: str) -> str by llm(stream=True, logging=True);

    events = stream_events(explain_jac("Jac language"));
    kinds = set(event_types(events));
    assert kinds == {"chunk", "usage"} , (
        f"the no-tools path yields only chunk and usage, got {sorted(kinds)}"
    );

    usage = events_of(events, "usage");
    assert len(usage) == 1 , f"expected exactly one usage event, got {len(usage)}";
    assert usage[0] is events[-1] , "usage is the terminal event";
    assert "total" in usage[0].data and "per_call" in usage[0].data , (
        f"usage carries total and per_call, got {sorted(usage[0].data)}"
    );

    text = chunk_text(events);
    assert "Jac" in text and "programming language" in text , f"got {text!r}";
}


test "a logging stream with tools reports the call, the result, then the answer" {
    answer = "Based on the information gathered, London is sunny at 25 degrees.";
    llm = FakeLLM(
        replies=[
            call("get_weather", {"city": "London"}),
            finish("The weather in London is sunny, 25 degrees."),
            say(answer)
        ]
    );

    """Get the current weather for a city."""
    def check_weather(question: str) -> str by llm(
        tools=[get_weather], stream=True, logging=True
    );

    events = stream_events(check_weather("What is the weather in London?"));

    calls = events_of(events, "tool_call");
    assert len(calls) >= 1 , f"expected a tool_call event, got {event_types(events)}";
    assert calls[0].data["tool"] == "get_weather" , f"got {calls[0].data}";

    results = events_of(events, "tool_result");
    assert len(results) >= 1 , f"expected a tool_result event, got {event_types(
        events
    )}";
    assert "sunny" in results[0].data["result"] , f"got {results[0].data['result']!r}";

    assert len(events_of(events, "steps_done")) >= 1 , (
        f"expected a steps_done event, got {event_types(events)}"
    );

    # Tool activity is reported before the answer starts arriving.
    kinds = event_types(events);
    last_tool = len(kinds) - 1 - kinds[::-1].index("tool_result");
    assert last_tool < kinds.index("chunk") , (
        f"tool events must precede chunks, got {kinds}"
    );

    assert "London" in chunk_text(events) , f"got {chunk_text(events)!r}";

    usage = events_of(events, "usage");
    assert len(usage) == 1 and usage[0] is events[-1] , (
        f"one terminal usage event, got {event_types(events)}"
    );
    assert "total" in usage[0].data and "per_call" in usage[0].data;
}


test "a str answer that skips finish_tool still arrives on the chunk channel" {
    # A tool is offered so the ReAct loop is entered, but the model answers in
    # plain text instead of calling finish_tool. The terminal text must reach
    # `chunk`, not be left as thought-only.
    llm = FakeLLM(replies=[say("It is sunny in London, about 25 degrees.")]);

    def answer(question: str) -> str by llm(
        tools=[get_weather], stream=True, logging=True
    );

    events = stream_events(answer("What is the weather in London?"));
    assert "sunny in London" in chunk_text(events) , (
        f"the answer must be on the chunk channel, got {chunk_text(events)!r}"
    );
}


test "a structured return cannot stream without a handler to consume it" {
    # Streaming a non-str return has no well-defined result unless something
    # consumes the events, so it is rejected up front rather than half-working.
    llm = FakeLLM(replies=[typed([1])]);

    def pick(options: str) -> list[int] by llm(stream=True);

    raised: any = None;
    try {
        pick("0=a, 1=b");
    } except ConfigurationError as e {
        raised = e;
    }
    assert raised is not None , (
        "structured streaming without a stream_handler must raise ConfigurationError"
    );
}


"""End-of-stream sentinel, so a worker can report it by identity."""
glob _STOP = object();

def _next_or_stop(gen: any) -> object {
    try {
        return next(gen);
    } except StopIteration {
        return _STOP;
    }
}


"""Drain a generator inside a fresh contextvars Context."""
def _drain_in_copied_context(gen: any) -> list {
    return contextvars.copy_context().run(list, gen);
}


"""Drive each next() on a thread-pool worker, as the scale backend's SSE path does."""
def _drain_in_thread_pool(gen: any) -> list {
    async def _run -> list {
        loop = asyncio.get_event_loop();
        out: list = [];
        while True {
            item = await loop.run_in_executor(None, _next_or_stop, gen);
            if item is _STOP {
                break;
            }
            out.append(item);
        }
        return out;
    }
    return asyncio.run(_run());
}


test "a stream survives being consumed from a foreign context" {
    # Regression for the scale backend's sse_frame_stream, which drives next()
    # on a thread-pool worker. The wrapper used to reset its invocation-id Token
    # in a finally that ran in that foreign Context, raising "Token was created
    # in a different Context" and killing the stream.
    for (label, drain) in [
        ("copied context", _drain_in_copied_context),
        ("thread pool", _drain_in_thread_pool)
    ] {
        answer = f"alpha beta gamma from a {label}";
        llm = FakeLLM(replies=[say(answer)]);

        def stream_answer -> str by llm(stream=True);

        chunks = drain(stream_answer());
        assert chunks , f"{label}: iteration produced no chunks";
        assert "".join(chunks) == answer , f"{label}: got {''.join(chunks)!r}";
        assert _current_invocation_id.get(None) is None , (
            f"{label}: the invocation id leaked into the caller's Context"
        );
    }
}
