"""Image and video inputs: how a media argument reaches the model.

Assert on the message blocks byLLM actually sent, not on a rendered string.
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


test "an image argument is inlined in its own format, under a system prompt" {
    need("PIL", "Pillow");
    # (file, expected data-url mime). A webp must not be transcoded to jpeg.
    for (filename, mime) in [
        ("math_question.jpg", "data:image/jpeg;base64,"),
        ("webp_image_of_person.webp", "data:image/webp;base64,")
    ] {
        llm = FakeLLM(replies=[say("answered")]);

        def describe(question_img: Image) -> str by llm();

        assert describe(Image(os.path.join(FIXTURE_DIR, filename))) == "answered";
        assert roles(llm) == ["system", "user"] , f"{filename}: got {roles(llm)}";

        blocks = user_blocks(llm);
        assert blocks[0]["type"] == "text" , f"{filename}: got {blocks[0]}";
        assert "describe" in blocks[0]["text"] , (
            f"{filename}: the caller name introduces the argument, got {blocks[0][
                'text'
            ][:60]!r}"
        );

        images = media_blocks(llm);
        assert len(images) == 1 , f"{filename}: expected one image block, got {len(
            images
        )}";
        assert data_url(images[0]).startswith(mime) , (
            f"{filename}: expected {mime}, got {data_url(images[0])[:40]!r}"
        );
    }
}


test "an image answer is parsed into its declared type" {
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
