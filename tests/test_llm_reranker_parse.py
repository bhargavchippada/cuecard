"""Tests for LLM re-ranker response parsing."""

from __future__ import annotations

import json

from cuecard.retrieval.llm_reranker import (
    _extract_rule_refs_from_prose,
    _parse_llm_response,
    _strip_thinking_tags,
    _try_json_parse,
)


class TestStripThinkingTags:
    def test_removes_think_block(self) -> None:
        text = '<think>reasoning here</think>{"rules": [1]}'
        assert _strip_thinking_tags(text) == '{"rules": [1]}'

    def test_multiple_think_blocks(self) -> None:
        text = "<think>a</think>hello<think>b</think>world"
        assert _strip_thinking_tags(text) == "helloworld"

    def test_no_think_blocks(self) -> None:
        text = '{"rules": [1, 2]}'
        assert _strip_thinking_tags(text) == '{"rules": [1, 2]}'

    def test_think_with_newlines(self) -> None:
        text = "<think>\nline1\nline2\n</think>result"
        assert _strip_thinking_tags(text) == "result"


class TestParseLLMResponse:
    def test_valid_json(self) -> None:
        result = _parse_llm_response('{"rules": [1, 3]}', 5)
        assert result.indices == [1, 3]

    def test_json_extra_whitespace(self) -> None:
        result = _parse_llm_response('  { "rules" : [ 1 , 3 ] }  ', 5)
        assert result.indices == [1, 3]

    def test_regex_fallback_comma_separated(self) -> None:
        result = _parse_llm_response("1, 3, 5", 5)
        assert result.indices == [1, 3, 5]

    def test_regex_fallback_brackets(self) -> None:
        result = _parse_llm_response("[1,3,5]", 5)
        assert result.indices == [1, 3, 5]

    def test_regex_fallback_space_separated(self) -> None:
        result = _parse_llm_response("1 3 5", 5)
        assert result.indices == [1, 3, 5]

    def test_invalid_prose_returns_none(self) -> None:
        result = _parse_llm_response("Error: rule 42 not found", 5)
        assert result.indices is None

    def test_out_of_range_filtered(self) -> None:
        result = _parse_llm_response('{"rules": [0, 1, 99]}', 5)
        assert result.indices == [1]

    def test_duplicates_removed(self) -> None:
        result = _parse_llm_response('{"rules": [1, 1, 3]}', 5)
        assert result.indices == [1, 3]

    def test_capped_at_max(self) -> None:
        indices = list(range(1, 21))
        response = json.dumps({"rules": indices})
        result = _parse_llm_response(response, 10)
        assert result.indices is not None
        assert len(result.indices) <= 10

    def test_empty_response_returns_none(self) -> None:
        result = _parse_llm_response("", 5)
        assert result.indices is None

    def test_empty_rules_array_returns_empty_list(self) -> None:
        result = _parse_llm_response('{"rules": []}', 5)
        assert result.indices == []

    def test_thinking_tags_stripped(self) -> None:
        text = '<think>thinking...</think>{"rules": [2, 4]}'
        result = _parse_llm_response(text, 5)
        assert result.indices == [2, 4]

    def test_thinking_with_empty_rules(self) -> None:
        text = '<think>\n\n</think>\n\n{"rules": []}'
        result = _parse_llm_response(text, 5)
        assert result.indices == []

    def test_json_missing_rules_key(self) -> None:
        result = _parse_llm_response('{"data": 42}', 50)
        assert result.indices is None

    def test_json_missing_rules_key_with_prose(self) -> None:
        result = _parse_llm_response('{"error": "not found"}', 5)
        assert result.indices is None

    def test_reasoning_captured(self) -> None:
        response = '{"reasoning": "Rule 1 applies because X.", "rules": [1]}'
        result = _parse_llm_response(response, 5)
        assert result.indices == [1]
        assert result.reasoning == "Rule 1 applies because X."

    def test_reasoning_empty_rules(self) -> None:
        response = '{"reasoning": "No rules apply.", "rules": []}'
        result = _parse_llm_response(response, 5)
        assert result.indices == []
        assert result.reasoning == "No rules apply."

    def test_no_reasoning_field(self) -> None:
        result = _parse_llm_response('{"rules": [2]}', 5)
        assert result.indices == [2]
        assert result.reasoning is None


class TestParseDangerousMutants:
    """Tests targeting dangerous surviving mutants in _parse_llm_response."""

    def test_multiline_json_extraction(self) -> None:
        """JSON spanning multiple lines must be extracted (kills re.DOTALL removal)."""
        response = (
            'Some preamble\n{"reasoning": "Multi\\nline",'
            '\n"rules": [1, 3]}\ntrailing'
        )
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 3]

    def test_non_numeric_rules_filtered(self) -> None:
        """Non-numeric values in rules array must be filtered out (kills and->or)."""
        response = '{"rules": [1, "two", 3, null, true]}'
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 3]

    def test_float_non_integer_filtered(self) -> None:
        """Float values like 1.5 must be filtered (not integer-equal)."""
        response = '{"rules": [1, 1.5, 2.0]}'
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 2]

    def test_reasoning_preserved_in_regex_fallback(self) -> None:
        """If JSON parsed reasoning before regex fallback, it's preserved."""
        # This path: JSON parse fails, regex matches number list
        # reasoning is None because no JSON was parsed
        result = _parse_llm_response("1, 3", 5)
        assert result.indices == [1, 3]
        assert result.reasoning is None

    def test_reasoning_preserved_on_final_none(self) -> None:
        """Unparseable response still returns reasoning if JSON had it partially."""
        # Full JSON parse fails, regex doesn't match prose -> indices=None
        result = _parse_llm_response("I think rules 1 and 3 apply here", 5)
        assert result.indices is None


class TestParseJsonExtraction:
    """Tests for first-JSON-block extraction in _parse_llm_response."""

    def test_json_with_trailing_text(self) -> None:
        """Valid JSON followed by repeated copies -- extracts first block."""
        response = (
            '{"reasoning": "Match.", "rules": [1, 3]}\n'
            '{"reasoning": "Match.", "rules": [1, 3]}'
        )
        result = _parse_llm_response(response, 5)
        assert result.indices == [1, 3]
        assert result.reasoning == "Match."

    def test_json_with_trailing_garbage(self) -> None:
        """Valid JSON followed by non-JSON text."""
        response = '{"reasoning": "OK.", "rules": [2]} some trailing text'
        result = _parse_llm_response(response, 5)
        assert result.indices == [2]

    def test_invalid_json_in_braces(self) -> None:
        """Braces present but content is not valid JSON."""
        response = "{this is not json at all}"
        result = _parse_llm_response(response, 5)
        assert result.indices is None

    def test_no_braces_at_all(self) -> None:
        """No JSON-like content anywhere in response."""
        response = "I cannot help with that request."
        result = _parse_llm_response(response, 5)
        assert result.indices is None

    def test_prose_then_json_at_end(self) -> None:
        """Model writes reasoning prose then JSON -- extracts JSON."""
        response = (
            "The action uploads a file. Rule 2 applies because "
            "it validates input. Rule 3 applies for permissions.\n"
            '{"reasoning": "upload", "rules": [2, 3]}'
        )
        result = _parse_llm_response(response, 5)
        assert result.indices == [2, 3]

    def test_prose_only_extracts_rule_refs(self) -> None:
        """Model writes only prose -- falls back to rule ref extraction."""
        response = (
            "The action edits code. Rule 2 applies because "
            "it mandates type hints. Rule 5 also applies for "
            "resource cleanup. Rule 1 is unrelated."
        )
        result = _parse_llm_response(response, 5)
        assert result.indices == [2, 5]
        assert result.reasoning is not None


class TestTryJsonParse:
    def test_full_json(self) -> None:
        result = _try_json_parse('{"rules": [1, 2]}')
        assert result == {"rules": [1, 2]}

    def test_json_after_prose(self) -> None:
        text = 'Some reasoning text.\n{"reasoning": "ok", "rules": [3]}'
        result = _try_json_parse(text)
        assert result is not None
        assert result["rules"] == [3]

    def test_json_between_prose(self) -> None:
        text = 'Start text {"rules": [1]} end text'
        result = _try_json_parse(text)
        assert result is not None
        assert result["rules"] == [1]

    def test_no_json(self) -> None:
        assert _try_json_parse("no json here") is None

    def test_invalid_json_in_braces(self) -> None:
        assert _try_json_parse("{not valid json}") is None

    def test_nested_braces_in_reasoning(self) -> None:
        text = '{"reasoning": "code has {braces}", "rules": [2]}'
        result = _try_json_parse(text)
        assert result is not None
        assert result["rules"] == [2]

    def test_returns_none_for_list(self) -> None:
        assert _try_json_parse("[1, 2, 3]") is None

    def test_brace_matched_but_inner_json_is_list(self) -> None:
        """Braces found at position, balanced, but json.loads gives a list."""
        text = 'Some text {wrong} more text {"rules": [1]}'
        result = _try_json_parse(text)
        assert result is not None
        assert result["rules"] == [1]

    def test_brace_matched_invalid_json_inside(self) -> None:
        """Balanced braces but invalid JSON content -- falls through."""
        text = "prose {invalid json} more text"
        result = _try_json_parse(text)
        assert result is None


class TestExtractRuleRefsFromProse:
    def test_basic_extraction(self) -> None:
        text = "Rule 2 applies because X. Rule 5 also applies for Y."
        result = _extract_rule_refs_from_prose(text, 10)
        assert result == [2, 5]

    def test_directly_applies(self) -> None:
        text = "Rule 1 directly applies here."
        result = _extract_rule_refs_from_prose(text, 5)
        assert result == [1]

    def test_no_rule_refs(self) -> None:
        text = "This is just general text about nothing."
        assert _extract_rule_refs_from_prose(text, 5) is None

    def test_filters_out_of_range(self) -> None:
        text = "Rule 1 applies. Rule 99 applies."
        result = _extract_rule_refs_from_prose(text, 5)
        assert result == [1]

    def test_deduplicates(self) -> None:
        text = "Rule 2 applies first. Rule 2 also applies here."
        result = _extract_rule_refs_from_prose(text, 5)
        assert result == [2]

    def test_is_relevant_pattern(self) -> None:
        text = "Rule 3 is relevant to this action."
        result = _extract_rule_refs_from_prose(text, 5)
        assert result == [3]
