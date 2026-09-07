"""What byLLM puts in the prompt, and what it makes of the reply.

sem strings, incl_info, temperature and receiver state on the way out; typed
returns, enums and scalars on the way back. Every test asserts on both
directions, which is the thing a fixture-and-stdout test could not do.
"""

import from support_tests { FakeLLM, call, prompt_text, roles, say, tool_names, typed }


enum _Sentiment { POSITIVE, NEGATIVE, NEUTRAL }


enum _LowerPersonality { INTROVERT = "Introvert", EXTROVERT = "Extrovert" }

obj _LowerPerson {
    has full_name: str,
        yod: int,
        personality: _LowerPersonality;
}


enum _MethodPersonality { INTROVERT, EXTROVERT }

obj _MethodPerson {
    has name: str,
        age: int;
}

glob _METHOD_PERSONALITIES: set[_MethodPersonality] = {
         _MethodPersonality.INTROVERT,
         _MethodPersonality.EXTROVERT
     },
     _sem_llm = FakeLLM(replies=[typed(120597), say("R8@jL3pQ")]);


def _generate_specific_number -> int by _sem_llm();

sem _generate_specific_number = "Generates a specific number that is 120597 and returns it.";

def _generate_password -> str by _sem_llm();

sem _generate_password = "Generates a password with an uppercase letter, a digit and a symbol.";


test "incl_info examples and temperature reach the model" {
    emoji_examples: list[dict[(str, str)]] = [
        {
            "input": "I love tp drink pina coladas",
            "output": "\U0001f464\u2764\ufe0f\U0001f964\U0001f34d\U0001f965"
        },
        {"input": "Mime Person", "output": "\U0001f464\U0001f932\U0001f6b7"}
    ];
    llm = FakeLLM(replies=[say("\U0001f464\u27a1\ufe0f\U0001f5fc")]);

    def get_emoji(input: str) -> str by llm(
        temperature=0.7, incl_info={"examples": emoji_examples}
    );

    assert get_emoji("Lets move to paris") == "\U0001f464\u27a1\ufe0f\U0001f5fc";
    sent = llm.seen[0];
    assert sent["temperature"] == 0.7 , f"temperature must reach the model, got {sent.get(
        'temperature'
    )}";
    assert "Mime Person" in str(sent["messages"]) , "incl_info examples must reach the prompt";
}


test "a method's own object reaches the prompt as context" {
    llm = FakeLLM(replies=[typed(86.75)]);

    obj Student {
        has name: str,
            marks: list[int] = [];

        def get_average -> float by llm();
    }
    student = Student(name="Alice", marks=[81, 99, 77, 90]);

    average = student.get_average();
    assert average == 86.75 , f"a float return must be converted, got {average!r}";
    assert isinstance(average, float) , f"expected float, got {type(average).__name__}";
    prompt = prompt_text(llm);
    assert "Alice" in prompt , "the receiver's fields must reach the prompt";
    assert "99" in prompt , f"the marks must reach the prompt: {prompt[:400]}";
}


test "a sem string reaches the prompt and a typed return is converted" {
    number = _generate_specific_number();
    assert number == 120597 , f"expected 120597, got {number!r}";
    assert isinstance(number, int) , (
        f"declared -> int, got {type(number).__name__}: parse_response did not run"
    );
    assert "120597" in prompt_text(_sem_llm) , (
        "the sem string is the prompt; without it the model is told nothing"
    );

    password = _generate_password();
    assert password == "R8@jL3pQ";
    assert "uppercase letter" in prompt_text(_sem_llm, 1) , (
        "each call carries its own sem string"
    );
}


test "a typed object return is parsed, enum member and all" {
    llm = FakeLLM(
        replies=[
            say(
                '{"schema_object_wrapper": {"full_name": "J. Robert Oppenheimer", '
                '"yod": 1967, "personality": "Introvert"}}'
            )
        ]
    );

    def get_person_info(name: str) -> _LowerPerson by llm();

    person = get_person_info("Oppenheimer");
    assert person.full_name == "J. Robert Oppenheimer" , f"got {person!r}";
    assert person.yod == 1967;
    assert person.personality == _LowerPersonality.INTROVERT , (
        f"enum member must be reconstructed, got {person.personality!r}"
    );
    assert "Oppenheimer" in prompt_text(llm) , "the argument must reach the prompt";
}


test "a nested method's incl_info and receiver state reach the prompt" {
    llm = FakeLLM(replies=[typed(_MethodPersonality.INTROVERT.value)]);

    obj _Diarist {
        has diary_entries: list[str];

        def get_personality(person: _MethodPerson) -> _MethodPersonality by llm(
            incl_info={
                "personality_examples": _METHOD_PERSONALITIES,
                "diary_entries": self.diary_entries
            }
        );
    }
    diarist = _Diarist(
        diary_entries=[
            "I won noble prize in Physics",
            "I am popular for my theory of relativity"
        ]
    );

    result = diarist.get_personality(_MethodPerson(name="Albert Einstein", age=76));
    assert result == _MethodPersonality.INTROVERT , f"got {result!r}";

    prompt = prompt_text(llm);
    assert "theory of relativity" in prompt , (
        f"incl_info from the receiver must reach the prompt: {prompt[:400]}"
    );
    assert "Albert Einstein" in prompt , "the argument must reach the prompt";
}


test "an enum return is reconstructed as the member" {
    llm = FakeLLM(replies=[typed(_Sentiment.POSITIVE.value)]);

    def classify(text: str) -> _Sentiment by llm();

    result = classify("I love this!");
    assert result == _Sentiment.POSITIVE , f"got {result!r}";
    assert isinstance(result, _Sentiment) , (
        f"expected a _Sentiment member, got {type(result).__name__}"
    );
    assert "I love this!" in prompt_text(llm);
}


def _wind_speed(city: str) -> str {
    return f"The current wind speed in {city} is about 18-22 km/h.";
}

sem _wind_speed = "Get the current wind speed of a given city.";


"""One tool-using turn, so the recorded params carry a real tools array and a
real tool result. `extra` is merged into the by-llm call params."""
def _wind_report(llm: FakeLLM, extra: dict) -> str {
    answer = "The current wind speed in Puttalam is about 18-22 km/h.";
    llm.replies.extend([call("_wind_speed", {"city": "Puttalam"}), say(answer)]);
    llm.postinit();

    def wind_report(city: str) -> str by llm(tools=[_wind_speed], **extra);

    return wind_report("Puttalam");
}


test "an outgoing request carries the model, messages, tools and response format" {
    llm = FakeLLM();
    _wind_report(llm, {});
    params = llm.seen[0];

    for key in ["model", "messages", "tools", "max_tokens"] {
        assert key in params , f"missing {key} in the outgoing params: {sorted(
            params
        )}";
    }
    assert "temperature" not in params , (
        "temperature must be omitted when the caller never set one, so the provider "
        "applies its own default (issue #7687)"
    );

    # byLLM appends finish_tool whenever the caller passes tools, so a typed
    # answer has a way home.
    assert tool_names(params) == ["_wind_speed", "finish_tool"] , f"got {tool_names(
        params
    )}";
    fn = params["tools"][0];
    assert fn["type"] == "function";
    assert "city" in fn["function"]["parameters"]["properties"] , (
        f"the tool signature must reach the model: {fn['function']['parameters']}"
    );
    assert "wind speed of a given city" in str(fn["function"]) , (
        "the tool's sem string is its description"
    );
}


test "the second turn replays the tool result back to the model" {
    llm = FakeLLM();
    _wind_report(llm, {});
    assert len(llm.seen) == 2 , f"expected a tool turn then an answer turn, got {len(
        llm.seen
    )}";

    assert roles(llm, 1)[:2] == ["system", "user"] , f"got {roles(llm, 1)}";
    tool_msgs = [
        m
        for m in llm.sent("messages")[1]
        if m.get("role") == "tool"
    ];
    assert len(tool_msgs) == 1 , f"expected one tool result, got {len(tool_msgs)}";
    assert tool_msgs[0]["content"]
        == ("The current wind speed in Puttalam is about 18-22 km/h.") , f"got {tool_msgs[
        0
    ]['content']!r}";
}


test "temperature is sent when the caller sets one" {
    llm = FakeLLM();
    _wind_report(llm, {"temperature": 0.3});
    assert llm.seen[0]["temperature"] == 0.3 , f"got {llm.seen[0].get(
        'temperature'
    )!r}";
}
