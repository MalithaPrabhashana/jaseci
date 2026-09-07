import os;
import from jaclang.byllm.lib { Image, Video }
import from support_tests { FakeLLM, say, typed }

glob FIXTURE_DIR: str = os.path.join(os.path.dirname(__file__), "fixtures");


enum _WebpPersonality { INTROVERT = "Introvert", EXTROVERT = "Extrovert" }

obj _WebpPerson {
    has full_name: str,
        year_of_death: str,
        personality: _WebpPersonality;
}


test "an image argument reaches the prompt as a data url" {
    try {
        import PIL;
    } except ImportError {
        testskip("This test requires Pillow to be installed");
    }
    answer = "The image shows a hot air balloon shaped like a heart.";
    llm = FakeLLM(replies=[say(answer)]);

    def explain_the_image(image: Image) -> str by llm();

    image = Image(os.path.join(FIXTURE_DIR, "image.jpg"));
    assert explain_the_image(image) == answer;
    prompt = str(llm.sent("messages")[0]);
    assert "data:image/jpeg;base64," in prompt , "the image must be inlined into the prompt";
}


test "an image parameter is sent as a system prompt plus a data url" {
    try {
        import PIL;
    } except ImportError {
        testskip("This test requires Pillow to be installed");
    }
    llm = FakeLLM(replies=[say("The answer to the math question is 42.")]);

    def solve_math_question(question_img: Image) -> str by llm();

    answer = solve_math_question(Image(os.path.join(FIXTURE_DIR, "math_question.jpg")));
    assert answer == "The answer to the math question is 42.";

    messages = llm.sent("messages")[0];
    assert messages[0]["role"] == "system" , f"first message should be system: {messages[
        0
    ]}";
    user_blocks = messages[1]["content"];
    assert user_blocks[0]["type"] == "text" , f"got {user_blocks[0]}";
    assert "solve_math_question" in user_blocks[0]["text"] , (
        "the caller name should introduce the argument"
    );
    image_block = user_blocks[-1];
    assert image_block["type"] == "image_url" , f"got {image_block}";
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,") , (
        "the image must be inlined as a jpeg data url"
    );
}


test "a webp image is accepted and its typed answer parsed" {
    try {
        import PIL;
    } except ImportError {
        testskip("This test requires Pillow to be installed");
    }
    llm = FakeLLM(
        replies=[
            typed(
                {
                    "full_name": "Albert Einstein",
                    "year_of_death": "1955",
                    "personality": _WebpPersonality.EXTROVERT.value
                }
            )
        ]
    );

    def get_person_info(image_content: Image) -> _WebpPerson by llm();

    person = get_person_info(
        Image(os.path.join(FIXTURE_DIR, "webp_image_of_person.webp"))
    );
    assert person.full_name == "Albert Einstein" , f"got {person!r}";
    assert person.year_of_death == "1955";
    assert person.personality == _WebpPersonality.EXTROVERT;

    blocks = llm.sent("messages")[0][1]["content"];
    images = [
        b
        for b in blocks
        if b.get("type") == "image_url"
    ];
    assert len(images) == 1 , f"expected one image block, got {len(images)}";
    url = images[0]["image_url"]["url"];
    assert url.startswith("data:image/webp;base64,") , f"a webp must stay webp: {url[:60]}";
}


test "a video parameter is sampled into image frames" {
    try {
        import cv2;
    } except ImportError {
        testskip("This test requires OpenCV to be installed");
    }
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

    blocks = llm.sent("messages")[0][1]["content"];
    frames = [
        b
        for b in blocks
        if b.get("type") == "image_url"
    ];
    assert len(frames) > 1 , f"a video should sample several frames, got {len(frames)}";
    assert frames[0]["image_url"]["url"].startswith("data:image/") , (
        "frames must be inlined as images"
    );
}
