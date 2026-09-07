"""Image and video inputs: how a media argument reaches the model.

Each test asserts on the message blocks byLLM actually sent, so a change in how
media is encoded fails here instead of passing on a substring of a log line.
"""

import os;
import from jaclang.byllm.lib { Image, Video }
import from support_tests {
    FakeLLM,
    data_url,
    media_blocks,
    need,
    roles,
    say,
    typed,
    user_blocks
}

glob FIXTURE_DIR: str = os.path.join(os.path.dirname(__file__), "fixtures");


enum Personality { INTROVERT = "Introvert", EXTROVERT = "Extrovert" }

obj Person {
    has full_name: str,
        year_of_death: str,
        personality: Personality;
}


test "an image argument is sent as a system prompt, a text block, and a data url" {
    need("PIL", "Pillow");
    llm = FakeLLM(replies=[say("The answer to the math question is 42.")]);

    def solve_math_question(question_img: Image) -> str by llm();

    answer = solve_math_question(Image(os.path.join(FIXTURE_DIR, "math_question.jpg")));
    assert answer == "The answer to the math question is 42.";
    assert roles(llm) == ["system", "user"] , (
        f"expected a system turn then a user turn, got {roles(llm)}"
    );

    blocks = user_blocks(llm);
    assert blocks[0]["type"] == "text" , f"the argument should be introduced by text: {blocks[
        0
    ]}";
    assert "solve_math_question" in blocks[0]["text"] , (
        f"the caller name introduces the argument, got {blocks[0]['text'][:80]!r}"
    );

    images = media_blocks(llm);
    assert len(images) == 1 , f"expected one image block, got {len(images)}";
    assert data_url(images[0]).startswith("data:image/jpeg;base64,") , (
        f"a jpeg must be inlined as a jpeg data url, got {data_url(images[0])[:60]!r}"
    );
}


test "a webp image keeps its own format and its typed answer is parsed" {
    need("PIL", "Pillow");
    llm = FakeLLM(
        replies=[
            typed(
                {
                    "full_name": "Albert Einstein",
                    "year_of_death": "1955",
                    "personality": Personality.EXTROVERT.value
                }
            )
        ]
    );

    def get_person_info(image_content: Image) -> Person by llm();

    person = get_person_info(
        Image(os.path.join(FIXTURE_DIR, "webp_image_of_person.webp"))
    );
    assert person.full_name == "Albert Einstein" , f"got {person!r}";
    assert person.year_of_death == "1955" , f"got {person!r}";
    assert person.personality == Personality.EXTROVERT , f"got {person.personality!r}";

    images = media_blocks(llm);
    assert len(images) == 1 , f"expected one image block, got {len(images)}";
    assert data_url(images[0]).startswith("data:image/webp;base64,") , (
        f"a webp must not be transcoded, got {data_url(images[0])[:60]!r}"
    );
}


test "a video argument is sampled into several inlined frames" {
    need("cv2", "OpenCV");
    explanation = (
        "The video features a large rabbit emerging from a burrow in a lush, "
        "green environment."
    );
    llm = FakeLLM(replies=[say(explanation)]);

    def explain_the_video(video: Video) -> str by llm();

    result = explain_the_video(
        Video(path=os.path.join(FIXTURE_DIR, "SampleVideo_1280x720_2mb.mp4"), fps=1)
    );
    assert result == explanation;

    frames = media_blocks(llm);
    assert len(frames) > 1 , (
        f"a video should sample several frames, got {len(frames)}: sampling is a no-op"
    );
    for (i, frame) in enumerate(frames) {
        assert data_url(frame).startswith("data:image/") , (
            f"frame {i} must be an inlined image, got {data_url(frame)[:60]!r}"
        );
    }
}
