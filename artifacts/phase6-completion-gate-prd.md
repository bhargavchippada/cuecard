# Phase 6: Completion Gate — Stop Hook PRD v1.2

**Author:** Turiya
**Date:** 2026-04-08
**Status:** DRAFT (R1 + R2 findings addressed)

## Objective

Build a completion-gate Stop hook that prevents early agent termination by:
1. Capturing the user's original request from the session transcript
2. Retrieving relevant completion-criteria rules
3. Surfacing the user's request + rules to the agent with `decision: "block"`
4. Letting the agent self-evaluate whether its work is complete
5. Blocking up to N times per turn (configurable 1-4, default 2) before allowing stop

### Success Criteria

- [ ] Stop hook reads `last_assistant_message` and extracts user prompts from `transcript_path`
- [ ] Rules retrieved against user prompt context (not generic "end_turn")
- [ ] `decision: "block"` returned with user prompt + rules as `reason`
- [ ] Configurable max_stop_blocks (1-4, default 2) with counter file tracking
- [ ] Reads last N user messages from transcript (not just the last one)
- [ ] Daemon fast-path supports Stop with full context (same code paths as inline)
- [ ] Latency < 500ms for Stop hook (no LLM call in the hook itself)
- [ ] E2E test: agent continues after Stop blocks, completes within max_stop_blocks
- [ ] All transcript and counter file operations pass security review (path validation, scrubbing, permissions)

## Problem Statement

The current Stop hook receives `stop_reason: "end_turn"` with no useful context. It retrieves rules against a near-empty query, gets poor matches, and passively injects them as `additionalContext` that the agent ignores (it already decided to stop).

**Real problem:** Agents frequently stop early — they complete 80% of a task and present results without finishing the remaining 20%. The user has to say "you're not done, keep going." This is the #1 frustration pattern.

**Root cause:** The agent has no systematic "did I finish?" check. It relies on its own judgment, which is optimistic and biased toward stopping.

## Architecture

### Data Flow

```
UserPromptSubmit
  ├─ cuecard retrieves rules for user prompt (existing behavior)
  └─ Reset stop counter for session (new user message = new task)

Agent works... (PreToolUse/PostToolUse fire as normal)

Stop event fires:
  ├─ Input: { last_assistant_message, transcript_path, session_id, stop_hook_active? }
  │
  ├─ Guard 1: if stop_gate == false in config → exit 0 (feature disabled)
  ├─ Guard 2: if stop_hook_active == true → exit 0 (optional, may not exist)
  ├─ Guard 3: read counter file → if count >= max_stop_blocks → exit 0 (allow stop)
  │
  ├─ Validate transcript_path (must resolve under ~/.claude/, isfile, .jsonl)
  ├─ Extract last N user messages from transcript (byte-budget tail, 64KB)
  ├─ Apply scrub_secrets() to extracted messages and last_assistant_message
  │
  ├─ Build query: "Stop: User asked: {scrubbed_messages[:400]} | Agent said: {scrubbed_response[:200]}"
  │
  ├─ Retrieve rules via pipeline (embedding-only, threshold 0.40, NO LLM reranker)
  │   (rules are OPTIONAL bonus context — gate fires regardless)
  │
  ├─ Increment counter file (atomic read-check-increment under fcntl.flock)
  │
  └─ Output: { decision: "block", reason: "<formatted>" }
      │
      └─ reason ALWAYS includes:
           - Last N user requests (scrubbed, chronological, nonce-delimited)
           - Agent's response summary (scrubbed, first 200 chars)
           - "Review your work against the original requests."
           OPTIONALLY includes (if rules matched):
           - Matched completion-criteria rules as checklist
           ALWAYS ends with:
           - "If all items are addressed, you may stop.
              If not, continue working on remaining items."

Agent continues... → Stop fires again:
  ├─ Guard 3: count < max_stop_blocks → block again with updated context
  │   OR
  └─ Guard 3: count >= max_stop_blocks → allow stop (cleanup counter file) (cleanup counter file)
```

### Key Design Decisions

**D1: Configurable block count (1-4, default 2).**
Track block count in a session-scoped counter file under `~/.cuecard/state/` (not `/tmp/` — avoids world-readable files, symlink attacks, and stale files on shared systems). Session ID derived from `transcript_path` stem (UUID portion of the JSONL filename) — no external session_id needed, no path traversal risk. Each Stop event increments the counter atomically (fcntl.flock). When counter >= `max_stop_blocks`, allow the stop and delete the counter file. Default of 2 means the agent gets two self-review passes. Configurable via `[hooks] max_stop_blocks = 2` in cuecard.toml, validated to range 1-4 in `_VALIDATORS`. Counter resets on each new UserPromptSubmit (new user message = new task).

**D2: No LLM call in Stop hook. Rules are optional.**
The hook must be fast (< 500ms). Use embedding-only retrieval (dense + sparse + RRF) with threshold 0.40. Rules are bonus context — the gate ALWAYS fires (Option A) regardless of whether rules match. The core value is reminding the agent of the user's original request, not the rules. If retrieval returns 0 results, the block still fires with just the user prompts + agent response. Log a warning if `stop_gate=true` but no rules index exists.

**D3: Query combines user prompt + agent response.**
Neither alone is sufficient. The user prompt says what was asked. The agent response says what was done. The gap between them is what the rules should address. Both are scrubbed via `scrub_secrets()` before inclusion. Query format: `"Stop: User asked: {scrubbed_prompts[:400]} | Agent said: {scrubbed_response[:200]}"`.

**D4: Extract last N user messages from transcript.**
The `transcript_path` JSONL contains the full session. Validate path resolves under `~/.claude/` before reading. Read using byte-budget tail (seek from EOF, read last 64KB, split into lines) — never load full file. Find last N `type: "user"` messages (default N=3). Filter system-reminder blocks by checking for `<system-reminder>` prefix specifically (not bare `<` which drops legitimate messages). Configurable via `[hooks] stop_user_messages = 3`.

**D5: Output format is flat JSON, not wrapped in hookSpecificOutput.**
Stop hooks use `{"decision": "block", "reason": "..."}` directly — NOT the `hookSpecificOutput` wrapper that PreToolUse uses. This is a Claude Code convention difference. The daemon must branch on event type to emit the correct format.

**D6: Embedding-only mode for Stop, hardcoded.**
Always embedding-only, not configurable. Higher threshold (0.40) compensates for missing LLM reranker. Threshold stored as constant `_STOP_THRESHOLD = 0.40` in the adapter module. If no rules match, the gate still fires (Option A) — rules are optional bonus context.

**D7: Affinity mask still applies.**
Stop event mask filters to workflow + both rules (not tool_use-only). This is correct — Stop is an audit event, not a tool event.

**D8: Security invariants (from R1 review).**
- `transcript_path` validated via `Path.resolve()` + allowlist (`~/.claude/`) before reading. Must be a regular file with `.jsonl` extension.
- Session ID derived from transcript_path stem (e.g., `00893aaf-19fa-41d2-8238-13269b9b3ca0`), sanitized to `[A-Za-z0-9_-]` only. Never from external input.
- Counter files written with `os.open(O_WRONLY | O_CREAT | O_NOFOLLOW, 0o600)` — no symlink following.
- Counter directory `~/.cuecard/state/` created with `0o700` permissions.
- All user messages and assistant responses scrubbed via `scrub_secrets()` before inclusion in query or reason.
- User messages in `reason` delimited with `[USER REQUEST N]:` framing to prevent prompt injection.
- `last_assistant_message` scrubbed before use in query and logging.

**D9: `stop_hook_active` is optional.**
Counter is the primary guard. If `stop_hook_active` field exists and is true, also allow (belt-and-suspenders). If the field doesn't exist in the Claude Code protocol, the counter alone handles loop prevention.

**D10: New module `src/cuecard/stop_gate.py` with shared functions (R2 fix).**
All Stop gate logic lives in `stop_gate.py` — imported by both `claude_code.py` and `serve.py`. Follows the `pipeline.py` pattern (shared orchestrator). Contains: `execute_stop_gate()`, `reset_stop_counter()`, `_check_and_increment()`, `_validate_transcript_path()`, `_extract_last_user_messages()`, `_tail_lines()`, `_derive_session_id()`, `_format_completion_check()`. The existing `_handle_stop` in `_EVENT_HANDLERS` is removed — Stop is fully handled by the dedicated branch before dispatch. The `reset_stop_counter()` is called from UserPromptSubmit handlers in BOTH inline adapter and daemon paths (same bypass-prevention principle). Concurrent Stop hooks are not expected — Claude Code dispatches hooks sequentially within a session. The flock serializes access; cross-session contention is prevented by session-scoped filenames.

**D11: Option A — always block, rules are bonus.**
The gate ALWAYS fires (up to max_stop_blocks) regardless of whether rule retrieval returns results. The core value is surfacing the user's original request and agent response for self-evaluation. Matched rules are optional supplementary context presented as a checklist. If `stop_gate=true` but no rules index exists, log a warning but still block with just the user prompt reminder.

### Stop Query Construction

```python
def _handle_stop(data: dict, config: ResolvedConfig) -> tuple[str, str, str]:
    transcript_path = str(data.get("transcript_path", ""))
    n_messages = config.stop_user_messages  # validated int, default 3

    # Validate transcript path (D8: security)
    validated_path = _validate_transcript_path(transcript_path)

    # Extract last N user messages for full context
    user_messages = _extract_last_user_messages(validated_path, n=n_messages)
    # Scrub secrets from user messages (H1: security)
    scrubbed_messages = [scrub_secrets(m) for m in user_messages]
    user_context = " | ".join(scrubbed_messages) if scrubbed_messages else ""

    # Get agent's final response, scrubbed (M3: security)
    raw_assistant = str(data.get("last_assistant_message", ""))[:500]
    assistant_msg = scrub_secrets(raw_assistant)

    # Combine for retrieval — user intent + agent response
    if user_context:
        query = f"Stop: User asked: {user_context[:400]} | Agent said: {assistant_msg[:200]}"
    else:
        query = f"Stop: {assistant_msg[:500]}"

    return query, "", "Stop"
```

### Transcript Path Validation

```python
_TRANSCRIPT_ALLOWED_ROOTS = (
    Path.home() / ".claude",
)

def _validate_transcript_path(path: str) -> str | None:
    """Validate transcript path resolves under allowed roots.

    Returns validated path string, or None if invalid.
    """
    if not path:
        return None

    resolved = Path(path).resolve()

    # Must be a regular file with .jsonl extension
    if not resolved.is_file() or resolved.suffix != ".jsonl":
        return None

    # Must be under an allowed root
    for root in _TRANSCRIPT_ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return str(resolved)
        except ValueError:
            continue

    logger.warning("transcript_path %s not under allowed roots, rejecting", path[:80])
    return None
```

### Transcript Extraction

```python
_TAIL_BYTES = 65536  # 64KB — enough for ~200 JSONL lines
_MAX_LINE_LEN = 4096  # Skip lines longer than 4KB (tool output dumps)
_SYSTEM_TAGS = ("<system-reminder>", "<local-command-", "<command-name>", "<task-notification>")

def _tail_lines(path: str, max_bytes: int = _TAIL_BYTES) -> list[str]:
    """Read last max_bytes from file, split into lines.

    Seeks from EOF — never loads full file into memory.
    Discards first (potentially partial) line.
    """
    file_size = os.path.getsize(path)
    read_start = max(0, file_size - max_bytes)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(read_start)
        raw = f.read(max_bytes)

    lines = raw.splitlines()
    # Discard first line (likely partial if we seeked mid-file)
    if read_start > 0 and lines:
        lines = lines[1:]
    # Filter overly long lines (tool output dumps)
    return [ln for ln in lines if len(ln) <= _MAX_LINE_LEN]


def _extract_last_user_messages(transcript_path: str | None, n: int = 3) -> list[str]:
    """Read transcript JSONL, find last N user messages.

    Returns messages in chronological order (oldest first).
    Skips system-reminder and hook-generated messages.
    """
    if not transcript_path:
        return []

    try:
        lines = _tail_lines(transcript_path)
    except OSError:
        return []

    messages: list[str] = []

    for line in reversed(lines):
        if len(messages) >= n:
            break
        try:
            entry = json.loads(line)
            if entry.get("type") != "user":
                continue
            content = entry.get("message", {}).get("content", "")
            text = ""
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        t = block["text"].strip()
                        # Skip known system-injected tags (specific, not broad)
                        if not any(t.startswith(tag) for tag in _SYSTEM_TAGS):
                            parts.append(t)
                text = " ".join(parts).strip()

            # Skip empty and very short messages (e.g., "y", "ok")
            if text and len(text) > 2:
                messages.append(text[:300])
        except (json.JSONDecodeError, KeyError):
            continue

    messages.reverse()  # Chronological order
    return messages
```

### Block Counter Management

```python
_STATE_DIR = Path.home() / ".cuecard" / "state"
_SESSION_ID_RE = re.compile(r"[^a-zA-Z0-9_-]")

def _derive_session_id(transcript_path: str | None) -> str:
    """Derive session ID from transcript path stem.

    E.g., ~/.claude/projects/.../00893aaf-19fa-41d2-8238-13269b9b3ca0.jsonl
    → "00893aaf-19fa-41d2-8238-13269b9b3ca0"
    Falls back to "unknown" if path is empty/invalid.
    """
    if not transcript_path:
        return "unknown"
    stem = Path(transcript_path).stem
    # Sanitize to safe chars only (prevent path traversal)
    sanitized = _SESSION_ID_RE.sub("", stem)[:128]
    return sanitized or "unknown"


def _get_counter_path(session_id: str) -> Path:
    return _STATE_DIR / f"cuecard-stop-{session_id}.count"


def _read_stop_count(session_id: str) -> int:
    path = _get_counter_path(session_id)
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError, OSError):
        return 0


def _check_and_increment(session_id: str, max_blocks: int) -> tuple[int, bool]:
    """Atomically check counter against max and increment if below.

    Returns (new_count, should_block).
    Merges guard check + increment into single locked operation (R2 fix).
    Uses os.read() directly — no os.dup() needed (R2: eliminates fd leak risk).
    """
    if session_id == "unknown":
        # No valid transcript path — skip gate entirely (R2: prevent collision)
        return 0, False

    _STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = _get_counter_path(session_id)

    fd = os.open(
        str(path),
        os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)

        # Read current count directly (no dup needed)
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 32)
        try:
            count = int(raw.strip()) + 1
        except (ValueError, EOFError):
            count = 1

        if count > max_blocks:
            # Max reached — cleanup and allow stop
            os.close(fd)
            fd = -1  # Prevent double-close in finally
            with contextlib.suppress(FileNotFoundError, OSError):
                Path(path).unlink()
            return count, False

        # Write new count
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(count).encode())
        return count, True
    finally:
        if fd >= 0:
            os.close(fd)  # Also releases flock


def _reset_stop_count(session_id: str) -> None:
    """Called from UserPromptSubmit handler — new user message resets counter."""
    path = _get_counter_path(session_id)
    with contextlib.suppress(FileNotFoundError, OSError):
        path.unlink()


def _cleanup_stop_count(session_id: str) -> None:
    """Called when max_stop_blocks reached — remove stale counter."""
    _reset_stop_count(session_id)
```

### Stop Hook Output Format

When blocking (always, up to max_stop_blocks — Option A):
```json
{
  "decision": "block",
  "reason": "COMPLETION CHECK (1/2) — Review your work against the original requests.\n\n[USER REQUEST 1]: \"Expand the golden fixtures for each hook and benchmark e2e\"\n[USER REQUEST 2]: \"Also check coverage for PostToolUse\"\n\n[YOUR RESPONSE]: \"I've expanded the fixtures for PreToolUse and UserPromptSubmit...\"\n\nRelevant completion criteria:\n- [ ] Validate every phase against real data before moving to the next\n- [ ] Establish baseline metrics before implementing any improvement\n\nIf all items in the user's requests are addressed, you may stop.\nIf not, continue working on the remaining items."
}
```

When blocking with no rules matched (still blocks — Option A):
```json
{
  "decision": "block",
  "reason": "COMPLETION CHECK (1/2) — Review your work against the original requests.\n\n[USER REQUEST 1]: \"Build auth module with tests\"\n\n[YOUR RESPONSE]: \"I've created the auth module...\"\n\nIf all items in the user's requests are addressed, you may stop.\nIf not, continue working on the remaining items."
}
```

When allowing (max_stop_blocks reached, feature disabled, or no valid transcript):
```
(exit 0, no output)
```

### Shared Stop Gate Function (D10)

Single function used by BOTH inline adapter and daemon — prevents bypass bugs:

```python
# In a new module: src/cuecard/stop_gate.py
# (or inline in claude_code.py — TBD at implementation)

_STOP_THRESHOLD = 0.40

def execute_stop_gate(
    data: dict[str, object],
    config: ResolvedConfig,
    index: Index | None,
    affinity: AffinityIndex | None,
    embedding_model: object | None,
) -> dict[str, object]:
    """Execute the completion gate. Returns block dict or empty dict (allow).

    Called by both inline adapter and daemon — single source of truth.
    """
    # Guard 1: feature disabled
    if not config.stop_gate:
        return {}

    # Guard 2: stop_hook_active (optional field, belt-and-suspenders)
    if data.get("stop_hook_active") is True:
        return {}

    # Validate transcript and derive session
    transcript_path = _validate_transcript_path(
        str(data.get("transcript_path", ""))
    )
    session_id = _derive_session_id(transcript_path)

    # Guard 3: atomic counter check + increment (merged, under flock)
    count, should_block = _check_and_increment(session_id, config.max_stop_blocks)
    if not should_block:
        return {}

    # Extract and scrub context
    user_messages = _extract_last_user_messages(
        transcript_path, n=config.stop_user_messages,
    )
    scrubbed_messages = [scrub_secrets(m) for m in user_messages]

    raw_assistant = str(data.get("last_assistant_message", ""))[:500]
    assistant_msg = scrub_secrets(raw_assistant)

    # Retrieve rules (optional bonus context — gate fires regardless per D11)
    rules_results = []
    if index is not None and index.size > 0 and embedding_model is not None:
        user_context = " | ".join(scrubbed_messages)
        query = f"Stop: User asked: {user_context[:400]} | Agent said: {assistant_msg[:200]}"

        from cuecard.pipeline import run_pipeline
        result = run_pipeline(
            query, index, config,
            embedding_model=embedding_model,
            mode="embedding",
            event="Stop", affinity=affinity,
        )
        rules_results = [r for r in result.results if r.score >= _STOP_THRESHOLD]
    else:
        logger.warning("stop_gate enabled but no rules index — blocking with prompt reminder only")

    # Format block response (always block — Option A)
    reason = _format_completion_check(
        scrubbed_messages, assistant_msg, rules_results, count, config.max_stop_blocks,
    )
    return {"decision": "block", "reason": reason}
```

### Daemon Support

The daemon branches on event type and calls the shared function:

```python
# serve.py — _process_request branches for Stop
def _process_request(data, index, config, affinity, embedding_model):
    event = _detect_event(data)

    if event == "Stop":
        # Stop uses flat output format (D5), not hookSpecificOutput
        return execute_stop_gate(data, config, index, affinity, embedding_model)

    # ... existing PreToolUse/PostToolUse/UserPromptSubmit/SubagentStart flow ...
    return _process_tool_request(data, index, config, affinity, embedding_model, event)
```

### Adapter main() Refactor

The inline adapter branches before dispatch:

```python
def main():
    # ... existing stdin parsing ...
    event = _detect_event(data)

    if event == "Stop":
        # Stop has its own output format (flat, not hookSpecificOutput)
        config = load_config(project_dir=Path.cwd())
        loaded = load_or_build(config, model)
        index = loaded.index if loaded else None
        affinity = loaded.affinity if loaded else None
        result = execute_stop_gate(data, config, index, affinity, model)
        if result:
            print(json.dumps(result))
        # No output = allow stop (exit 0)
        return

    # ... existing PreToolUse/PostToolUse/UserPromptSubmit/SubagentStart flow ...
    # Remove Stop from _EVENT_HANDLERS (fully handled above)
```

## Configuration

```toml
[hooks]
# Enable completion gate (block agent at Stop until work is verified)
stop_gate = false               # default: false (opt-in until proven reliable)

# Max times to block per user message (1-4)
max_stop_blocks = 2             # default: 2

# Number of recent user messages to include as context
stop_user_messages = 3          # default: 3
```

Config wiring required:
- Add `stop_gate: bool`, `max_stop_blocks: int`, `stop_user_messages: int` to `ResolvedConfig`
- Add to `_extract_flat()` to read from `[hooks]` section
- Add to `_DEFAULTS`: `"stop_gate": False, "max_stop_blocks": 2, "stop_user_messages": 3`
- Add validators: `"max_stop_blocks": (int, 1, 4)`, `"stop_user_messages": (int, 1, 10)`
- `stop_gate` validated as bool
- `stop_gate` defaults to `false` (opt-in until proven reliable)

## Fixture Impact

Current Stop fixtures are **unrealistic** — they have rich human-authored narratives as queries:
> `"Stop: Committed changes with git commit and pushed to feature branch"`

In reality, the query will be:
> `"Stop: User asked: expand fixtures and benchmark | Agent said: I've completed the fixture expansion..."`

**Action:** Rewrite all 77 Stop fixtures to match the new query format. Each fixture needs:
- `user_prompt` field (what the user asked)
- `assistant_response` field (what the agent said)
- Query constructed as `"Stop: User asked: {prompt} | Agent said: {response}"`

Preserve old fixtures for regression comparison. Run 3 verification rounds on new fixtures (session 24 lesson).

## Implementation Phases

### Phase 1: Config + Counter Infrastructure (2 files)
- Add `stop_gate`, `max_stop_blocks`, `stop_user_messages` to `config.py` (`ResolvedConfig`, `_extract_flat`, `_VALIDATORS`)
- Add `~/.cuecard/state/` directory management
- Implement atomic counter (fcntl.flock, O_NOFOLLOW, 0o600)
- Implement `_derive_session_id()` from transcript_path

### Phase 2: Transcript Reader (1 file, new module)
- Implement `_validate_transcript_path()` with allowlist
- Implement `_tail_lines()` with byte-budget (64KB seek from EOF)
- Implement `_extract_last_user_messages()` with system-tag filtering
- All paths scrubbed via `scrub_secrets()`

### Phase 3: Adapter Changes (1 file)
- Branch `main()` for Stop events (different output format)
- Wire `_handle_stop_main()` with guards, counter, retrieval, formatting
- Counter reset in UserPromptSubmit handler
- Force embedding-only mode with threshold 0.40

### Phase 4: Daemon Support (1 file)
- Branch `_process_request()` for Stop events
- Mirror ALL adapter logic (counter, guards, validation, scrubbing)
- Different output format (flat JSON vs hookSpecificOutput)
- Test daemon path separately to prevent bypass bugs

### Phase 5: Hook Registration
- Register Stop hook in `cuecard hook install`
- Ensure clean exit when feature disabled or counter exceeded

### Phase 6: Fixture Rewrite + Benchmark
- Rewrite 77 Stop fixtures with user_prompt + assistant_response
- Preserve old fixtures for comparison
- Run 3 verification rounds
- Benchmark with embedding-only, threshold 0.40
- Add completion-criteria rules to global.txt if needed

## Test Plan

### Unit Tests
- `_validate_transcript_path()` — valid path under ~/.claude/, path outside root (reject), symlink (reject), non-.jsonl (reject), directory (reject), nonexistent (reject)
- `_tail_lines()` — small file (< 64KB, full read), large file (seek from EOF), empty file, file with very long lines (> 4KB, filtered)
- `_extract_last_user_messages()` — valid transcript with N messages, transcript with < N messages, transcript with only system-reminders, transcript with list-of-blocks format, corrupt JSONL lines (skipped), empty transcript
- `_derive_session_id()` — UUID stem extraction, sanitization of special chars, empty path, max length enforcement
- `_increment_stop_count()` — first increment (file created), subsequent increments, corrupt counter (reset to 1), file permissions are 0o600
- `_reset_stop_count()` — removes file, idempotent on missing file
- `_handle_stop_main()` — stop_gate=false (allow), stop_hook_active=true (allow), count >= max (allow + cleanup), count < max + rules match (block), count < max + no rules (allow)
- Counter file does NOT follow symlinks (O_NOFOLLOW)
- `max_stop_blocks=0` in config → `ConfigError` (validated 1-4)
- `max_stop_blocks=1` → exactly 1 block then allow
- Secrets in user messages are scrubbed in reason output
- Secrets in assistant message are scrubbed in query and reason
- User messages with `<system-reminder>` prefix are filtered, but messages starting with `<div>` are kept
- Stop output format is flat `{"decision": "block", "reason": "..."}` (not hookSpecificOutput)

### Integration Tests
- Full Stop pipeline: transcript → validate → extract → scrub → retrieve → format → output
- Daemon Stop path exercises same code as inline path (anti-bypass test)
- Daemon returns flat JSON for Stop, hookSpecificOutput for PreToolUse
- Counter survives across multiple Stop calls in same session
- Counter resets on UserPromptSubmit

### E2E Manual Test
- Start Claude Code session with `stop_gate = true`
- Give a multi-step task
- Verify agent gets blocked at first Stop with completion criteria
- Verify agent continues working
- Verify agent gets blocked again if max_stop_blocks > 1
- Verify agent is allowed to stop after max_stop_blocks reached
- Verify counter file is cleaned up

## Risks

1. **Transcript parsing brittleness**: JSONL format may change between Claude Code versions. Mitigation: defensive parsing, graceful fallback to empty prompt, all JSONDecodeError silently skipped.

2. **Large transcripts**: Long sessions → large JSONL files. Mitigation: byte-budget tail (64KB from EOF, never load full file), per-line length cap (4KB).

3. **Concurrent transcript write**: Stop hook fires while Claude Code writes the transcript. Mitigation: discard first line after seek (likely partial), JSONDecodeError on partial lines is silently skipped.

4. **Block fatigue**: If rules always match, agent wastes turns on self-review. Mitigation: higher threshold (0.40), `stop_gate = false` default, max_stop_blocks cap.

5. **Latency**: Transcript reading + retrieval must stay < 500ms. Mitigation: embedding-only mode (no LLM), byte-budget tail, no config override to LLM mode.

6. **Stale counter files**: Session crashes leave counter files. Mitigation: counter under `~/.cuecard/state/` (not /tmp), cleanup on max reached, UserPromptSubmit resets. Could add TTL (1 hour) in future.

## Open Questions (All Resolved — R1 + R2)

### Resolved from R1
1. ~~session_id undefined~~ Derive from transcript_path stem (UUID).
2. ~~stop_hook_active unverified~~ Counter is primary guard. stop_hook_active is optional.
3. ~~Stop output format~~ Confirmed flat `{"decision": "block"}`. Verify empirically in Phase 5.
4. ~~Counter in /tmp~~ Moved to `~/.cuecard/state/` with 0o700/0o600, O_NOFOLLOW.
5. ~~startswith("<") filter~~ Specific `_SYSTEM_TAGS` list.
6. ~~Config fields~~ Wired: ResolvedConfig + _extract_flat + _VALIDATORS + _DEFAULTS.
7. ~~Secrets in reason~~ scrub_secrets() on all content.
8. ~~Daemon bypass~~ Shared `execute_stop_gate()` function (D10).
9. ~~Default aggressive~~ `stop_gate = false` (opt-in).
10. ~~stop_retrieval_mode~~ Removed. Always embedding-only.

### Resolved from R2
11. ~~os.dup fragile~~ Replaced with direct `os.read(fd, 32)`.
12. ~~No rules = silent no-op~~ Option A: always block. Rules are bonus context (D11).
13. ~~Reset/increment race~~ Documented: hooks are sequential. Merged guard+increment under flock.
14. ~~embedding_model not threaded~~ Shared function takes `embedding_model` param explicitly.
15. ~~_DEFAULTS missing~~ Added: `stop_gate: False, max_stop_blocks: 2, stop_user_messages: 3`.
16. ~~_handle_stop name collision~~ Removed Stop from `_EVENT_HANDLERS`. Dedicated branch + shared function.
17. ~~"unknown" session collision~~ If no valid transcript, skip gate entirely (return allow).
18. ~~Counter guard outside lock~~ Merged into `_check_and_increment()` — single atomic operation.
19. ~~stop_user_messages cap~~ Raised to `(int, 1, 10)`.
20. ~~Threshold location~~ Constant `_STOP_THRESHOLD = 0.40` in stop_gate module.
