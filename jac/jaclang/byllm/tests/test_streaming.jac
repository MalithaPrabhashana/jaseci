"""Streaming: what a `by llm(stream=True)` call hands back, chunk by chunk."""

import from support_tests { FakeLLM, say }


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
