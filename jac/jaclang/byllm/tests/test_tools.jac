"""Tool use and the ReAct loop: finish_tool, and recovering when a model skips it."""

import from support_tests { FakeLLM, finish, say }


"""A chat message from one person."""
obj Chat {
    has person: str,
        message: str;
}


"""A side-effect tool. Its presence is what causes finish_tool to be auto-added."""
def get_mood(character: str) -> str {
    return "cheerful";
}


test "plain text instead of finish_tool is re-prompted, not accepted" {
    # The gpt-4o failure mode: the model answers conversationally rather than
    # calling finish_tool. invoke should notice, re-prompt with finish_tool
    # forced, and extract the typed result from the second turn.
    line = "Greetings traveller, I have wares to trade!";
    llm = FakeLLM(replies=[say(line), finish({"person": "Merchant", "message": line})]);

    """Generate the NPC's opening line given a player name."""
    def generate_npc_line(player_name: str, npc_name: str) -> Chat by llm(
        tools=[get_mood]
    );

    result = generate_npc_line("Elara", "Merchant");
    assert result.person == "Merchant" , f"got {result!r}";
    assert result.message == line , f"got {result!r}";

    assert len(llm.seen) == 2 , (
        f"expected the plain answer then a recovery turn, got {len(llm.seen)}"
    );
    assert llm.exhausted() , "both scripted turns should have been consumed";
}
