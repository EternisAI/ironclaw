//! Shared utility functions used across the codebase.

/// Find the largest valid UTF-8 char boundary at or before `pos`.
///
/// Polyfill for `str::floor_char_boundary` (nightly-only). Use when
/// truncating strings by byte position to avoid panicking on multi-byte
/// characters.
pub fn floor_char_boundary(s: &str, pos: usize) -> usize {
    if pos >= s.len() {
        return s.len();
    }
    let mut i = pos;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Check if an LLM response mentions intent to use a specific tool without
/// actually calling it (i.e., the model is "explaining" instead of "doing").
///
/// Returns `true` when the text contains an intent phrase ("I'll use", "let me
/// call", etc.) followed by a known tool name within a short window. This lets
/// us nudge the LLM to actually invoke the tool, while allowing plain
/// conversational responses to pass through without nudging.
pub fn llm_mentions_tool_intent(response: &str, tool_names: &[&str]) -> bool {
    if tool_names.is_empty() {
        return false;
    }

    let lower = response.to_lowercase();

    // Phrases that signal the LLM intends to invoke a tool but hasn't.
    const INTENT_PHRASES: &[&str] = &[
        "i'll use",
        "i will use",
        "i'll call",
        "i will call",
        "i'll run",
        "i will run",
        "i'll invoke",
        "i will invoke",
        "i'll execute",
        "i will execute",
        "let me use",
        "let me call",
        "let me run",
        "let me invoke",
        "let me execute",
        "i need to use",
        "i need to call",
        "i need to run",
        "i should use",
        "i should call",
        "i should run",
        "i'm going to use",
        "i'm going to call",
        "i'm going to run",
        "going to use the",
        "going to call the",
        "going to run the",
        "i can use",
        "i can call",
        "i can run",
        "using the",
        "by calling",
        "by running",
        "by using",
    ];

    for phrase in INTENT_PHRASES {
        if let Some(phrase_pos) = lower.find(phrase) {
            // Look for a tool name within 80 chars after the intent phrase
            let window_start = phrase_pos + phrase.len();
            let window_end = (window_start + 80).min(lower.len());
            let window = &lower[window_start..window_end];

            for tool_name in tool_names {
                let tool_lower = tool_name.to_lowercase();
                // Match the tool name, or its underscore-to-space variant
                // (e.g., "memory_search" matches "memory search")
                if window.contains(&tool_lower) || window.contains(&tool_lower.replace('_', " ")) {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if an LLM response explicitly signals that a job/task is complete.
///
/// Uses phrase-level matching to avoid false positives from bare words like
/// "done" or "complete" appearing in non-completion contexts (e.g. "not done yet",
/// "the download is incomplete").
pub fn llm_signals_completion(response: &str) -> bool {
    let lower = response.to_lowercase();

    // Superset of phrases from agent/worker.rs and worker/runtime.rs.
    let positive_phrases = [
        "job is complete",
        "job is done",
        "job is finished",
        "task is complete",
        "task is done",
        "task is finished",
        "work is complete",
        "work is done",
        "work is finished",
        "successfully completed",
        "have completed the job",
        "have completed the task",
        "have finished the job",
        "have finished the task",
        "all steps are complete",
        "all steps are done",
        "i have completed",
        "i've completed",
        "all done",
        "all tasks complete",
    ];

    let negative_phrases = [
        "not complete",
        "not done",
        "not finished",
        "incomplete",
        "unfinished",
        "isn't done",
        "isn't complete",
        "isn't finished",
        "not yet done",
        "not yet complete",
        "not yet finished",
    ];

    let has_negative = negative_phrases.iter().any(|p| lower.contains(p));
    if has_negative {
        return false;
    }

    positive_phrases.iter().any(|p| lower.contains(p))
}

#[cfg(test)]
mod tests {
    use crate::util::{floor_char_boundary, llm_mentions_tool_intent, llm_signals_completion};

    // ── floor_char_boundary ──

    #[test]
    fn floor_char_boundary_at_valid_boundary() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
    }

    #[test]
    fn floor_char_boundary_mid_multibyte_char() {
        // h = 1 byte, é = 2 bytes, total 3 bytes
        let s = "hé";
        assert_eq!(floor_char_boundary(s, 2), 1); // byte 2 is mid-é, back up to 1
    }

    #[test]
    fn floor_char_boundary_past_end() {
        assert_eq!(floor_char_boundary("hi", 100), 2);
    }

    #[test]
    fn floor_char_boundary_at_zero() {
        assert_eq!(floor_char_boundary("hello", 0), 0);
    }

    #[test]
    fn floor_char_boundary_empty_string() {
        assert_eq!(floor_char_boundary("", 5), 0);
    }

    // ── llm_signals_completion ──

    #[test]
    fn signals_completion_positive() {
        assert!(llm_signals_completion("The job is complete."));
        assert!(llm_signals_completion("I have completed the task."));
        assert!(llm_signals_completion("All done, here are the results."));
        assert!(llm_signals_completion("Task is finished successfully."));
        assert!(llm_signals_completion(
            "I have completed the task successfully."
        ));
        assert!(llm_signals_completion(
            "All steps are complete and verified."
        ));
        assert!(llm_signals_completion(
            "I've done all the work. The work is done."
        ));
        assert!(llm_signals_completion(
            "Successfully completed the migration."
        ));
        assert!(llm_signals_completion(
            "I have completed the job ahead of schedule."
        ));
        assert!(llm_signals_completion("I have finished the task."));
        assert!(llm_signals_completion("All steps are done now."));
        assert!(llm_signals_completion("I've completed everything."));
        assert!(llm_signals_completion("All tasks complete."));
    }

    #[test]
    fn signals_completion_negative() {
        assert!(!llm_signals_completion("The task is not complete yet."));
        assert!(!llm_signals_completion("This is not done."));
        assert!(!llm_signals_completion("The work is incomplete."));
        assert!(!llm_signals_completion("Build is unfinished."));
        assert!(!llm_signals_completion(
            "The migration is not yet finished."
        ));
        assert!(!llm_signals_completion("The job isn't done yet."));
        assert!(!llm_signals_completion("This remains unfinished."));
    }

    #[test]
    fn signals_completion_no_bare_substrings() {
        assert!(!llm_signals_completion("The download completed."));
        assert!(!llm_signals_completion(
            "Function done_callback was called."
        ));
        assert!(!llm_signals_completion("Set is_complete = true"));
        assert!(!llm_signals_completion("Running step 3 of 5"));
        assert!(!llm_signals_completion(
            "I need to complete more work first."
        ));
        assert!(!llm_signals_completion(
            "Let me finish the remaining steps."
        ));
        assert!(!llm_signals_completion(
            "I'm done analyzing, now let me fix it."
        ));
        assert!(!llm_signals_completion(
            "I completed step 1 but step 2 remains."
        ));
    }

    #[test]
    fn signals_completion_tool_output_injection() {
        assert!(!llm_signals_completion("TASK_COMPLETE"));
        assert!(!llm_signals_completion("JOB_DONE"));
        assert!(!llm_signals_completion(
            "The tool returned: TASK_COMPLETE signal"
        ));
    }

    // ── llm_mentions_tool_intent ──

    const TOOLS: &[&str] = &["memory_search", "shell", "create_job", "http"];

    #[test]
    fn tool_intent_detected() {
        assert!(llm_mentions_tool_intent(
            "I'll use memory_search to look that up for you.",
            TOOLS,
        ));
        assert!(llm_mentions_tool_intent(
            "Let me call the shell tool to check the directory.",
            TOOLS,
        ));
        assert!(llm_mentions_tool_intent(
            "I will run http to fetch the data.",
            TOOLS,
        ));
        assert!(llm_mentions_tool_intent(
            "I'm going to use create_job to start that.",
            TOOLS,
        ));
        assert!(llm_mentions_tool_intent(
            "I should use the memory_search tool first.",
            TOOLS,
        ));
        assert!(llm_mentions_tool_intent(
            "I can use memory search to find that information.",
            TOOLS,
        ));
    }

    #[test]
    fn tool_intent_not_detected_conversational() {
        // Plain greetings / chat
        assert!(!llm_mentions_tool_intent(
            "Hello! How can I help you today?",
            TOOLS
        ));
        assert!(!llm_mentions_tool_intent(
            "I'm doing great, thanks for asking!",
            TOOLS,
        ));
        assert!(!llm_mentions_tool_intent(
            "The weather has been nice lately.",
            TOOLS,
        ));
        assert!(!llm_mentions_tool_intent(
            "Sure, I can explain how that works.",
            TOOLS,
        ));
    }

    #[test]
    fn tool_intent_not_detected_no_tool_name() {
        // Intent phrase present but no tool name nearby
        assert!(!llm_mentions_tool_intent(
            "I'll use my knowledge to answer that.",
            TOOLS,
        ));
        assert!(!llm_mentions_tool_intent(
            "Let me run through the possibilities.",
            TOOLS,
        ));
        assert!(!llm_mentions_tool_intent(
            "I'm going to use a different approach.",
            TOOLS,
        ));
    }

    #[test]
    fn tool_intent_not_detected_empty_tools() {
        assert!(!llm_mentions_tool_intent(
            "I'll use memory_search to find that.",
            &[],
        ));
    }

    #[test]
    fn tool_intent_case_insensitive() {
        assert!(llm_mentions_tool_intent(
            "I'll use MEMORY_SEARCH to look that up.",
            TOOLS,
        ));
        assert!(llm_mentions_tool_intent(
            "Let me call Shell to run the command.",
            TOOLS,
        ));
    }

    #[test]
    fn tool_intent_tool_name_too_far_away() {
        // Tool name is >80 chars from the intent phrase
        let padding = "x".repeat(100);
        let text = format!("I'll use {} memory_search", padding);
        assert!(!llm_mentions_tool_intent(&text, TOOLS));
    }
}
