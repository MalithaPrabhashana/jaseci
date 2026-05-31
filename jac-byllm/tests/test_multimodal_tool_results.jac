"""Tests for multimodal tool result support (byllm/types.jac + message.impl.jac).

One test per helper added for the fix.  Each proves the *new* behavior
(Media survives the path); back-compat for string-returning tools is
covered by the existing test_parallel.jac suite.
"""

import from byllm.types {
    ToolCallResultMsg,
    Text,
    Image,
    normalize_tool_result,
    content_preview,
    reroute_tool_images_to_user_message
}


test "normalize_tool_result preserves list[Media]" {
    items = [Text(text="ok"), Image(url="https://example.com/x.png")];
    out = normalize_tool_result(items);
    assert out == items;
}


test "ToolCallResultMsg.to_dict serializes Media to image_url block" {
    msg = ToolCallResultMsg(
        content=[Text(text="PASS"), Image(url="https://example.com/x.png")],
        tool_call_id="call_1",
        name="browser_validate"
    );
    blocks = msg.to_dict()["content"];
    assert blocks[0] == {"type": "text", "text": "PASS"};
    assert blocks[1]["type"] == "image_url";
    assert blocks[1]["image_url"]["url"] == "https://example.com/x.png";
}


test "content_preview yields text without leaking base64" {
    preview = content_preview(
        [Text(text="PASS"), Image(url="data:image/png;base64,iVBORw0KGgoLONGBLOB")], 500
    );
    assert "PASS" in preview;
    assert "[image]" in preview;
    assert "iVBORw0KGgo" not in preview;
}


test "reroute_tool_images_to_user_message moves tool image into a following user message" {
    messages = [
        {"role": "user", "content": "look"},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "browser_validate",
            "content": [
                {"type": "text", "text": "PASS"},
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]
        }
    ];
    out = reroute_tool_images_to_user_message(messages);
    # tool message survives with text only (image stripped) — tool_call_id intact
    assert out[1]["role"] == "tool";
    assert out[1]["tool_call_id"] == "call_1";
    assert out[1]["content"] == "PASS";
    # image relocated into a new user message right after the tool message
    assert out[2]["role"] == "user";
    assert out[2]["content"][1]["type"] == "image_url";
    assert out[2]["content"][1]["image_url"]["url"] == "https://example.com/x.png";
}
