use super::*;

#[test]
fn check_messages_at_limit_passes() {
    assert!(check_messages(MAX_MESSAGES).is_ok());
}

#[test]
fn check_messages_over_limit_fails() {
    let err = check_messages(MAX_MESSAGES + 1).unwrap_err();
    assert!(matches!(
        err,
        BoundError::ItemsExceeded {
            field: "messages",
            ..
        }
    ));
}

#[test]
fn check_tool_calls_at_limit_passes() {
    assert!(check_tool_calls(MAX_TOOL_CALLS, 0).is_ok());
}

#[test]
fn check_tool_calls_over_limit_fails() {
    let err = check_tool_calls(MAX_TOOL_CALLS + 1, 0).unwrap_err();
    assert!(matches!(
        err,
        BoundError::ItemsExceeded {
            field: "tool_calls",
            ..
        }
    ));
}

#[test]
fn check_tools_at_limit_passes() {
    assert!(check_tools(MAX_TOOLS).is_ok());
}

#[test]
fn check_tools_over_limit_fails() {
    let err = check_tools(MAX_TOOLS + 1).unwrap_err();
    assert!(matches!(
        err,
        BoundError::ItemsExceeded { field: "tools", .. }
    ));
}

#[test]
fn check_content_parts_at_limit_passes() {
    assert!(check_content_parts(MAX_CONTENT_PARTS, 0).is_ok());
}

#[test]
fn check_content_parts_over_limit_fails() {
    let err = check_content_parts(MAX_CONTENT_PARTS + 1, 0).unwrap_err();
    assert!(matches!(
        err,
        BoundError::ItemsExceeded {
            field: "content_parts",
            ..
        }
    ));
}

#[test]
fn check_input_audio_bytes_at_limit_passes() {
    assert!(check_input_audio_bytes(MAX_INPUT_AUDIO_BYTES, 0).is_ok());
}

#[test]
fn check_input_audio_bytes_over_limit_fails() {
    let err = check_input_audio_bytes(MAX_INPUT_AUDIO_BYTES + 1, 0).unwrap_err();
    assert!(matches!(
        err,
        BoundError::BytesExceeded {
            field: "input_audio",
            ..
        }
    ));
}

#[test]
fn check_total_input_tokens_estimate_at_limit_passes() {
    // bytes that produce exactly MAX estimate: MAX_TOTAL * 3 bytes → estimate = MAX_TOTAL
    assert!(check_total_input_tokens_estimate(MAX_TOTAL_INPUT_TOKENS_ESTIMATE * 3).is_ok());
}

#[test]
fn check_total_input_tokens_estimate_over_limit_fails() {
    // bytes that produce estimate = MAX + 1: (MAX + 1) * 3 bytes
    let bytes = (MAX_TOTAL_INPUT_TOKENS_ESTIMATE + 1) * 3;
    let err = check_total_input_tokens_estimate(bytes).unwrap_err();
    assert!(matches!(
        err,
        BoundError::LimitExceeded {
            field: "total_input_tokens_estimate",
            ..
        }
    ));
}

#[test]
fn bound_error_message_contains_field_and_max() {
    let err = BoundError::ItemsExceeded {
        field: "messages",
        got: 5000,
        max: MAX_MESSAGES,
    };
    let msg = err.to_string();
    assert!(msg.contains("messages"));
    assert!(msg.contains("4096"));
}
