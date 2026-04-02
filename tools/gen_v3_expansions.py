"""Generate v3 expansions for both corpora."""
import time
from pathlib import Path
from cuecard.parser import parse_rules
from cuecard.expander import expand_rules
from cuecard.indexer import save_rules_json

# Basic corpus (PreToolUse rules)
basic_rules = parse_rules(("eval/corpora/rules_basic.txt",))
print(f"Expanding {len(basic_rules)} basic rules with v3 prompt (PreToolUse)...")
t0 = time.monotonic()
basic_expanded = expand_rules(
    basic_rules, backend="local", endpoint="http://localhost:8081/v1",
    event_type="PreToolUse",
)
print(f"Done in {time.monotonic()-t0:.1f}s")

out = "eval/corpora/enriched_basic_v3"
Path(out).mkdir(parents=True, exist_ok=True)
save_rules_json(basic_expanded, out)
total = sum(len(r.expansions) for r in basic_expanded)
print(f"Basic v3: {len(basic_expanded)} rules, {total} expansions (avg {total/len(basic_expanded):.1f})")

# Workflow corpus (UserPromptSubmit rules)
workflow_rules = parse_rules(("eval/corpora/rules_workflow.txt",))
print(f"\nExpanding {len(workflow_rules)} workflow rules with v3 prompt (UserPromptSubmit)...")
t0 = time.monotonic()
workflow_expanded = expand_rules(
    workflow_rules, backend="local", endpoint="http://localhost:8081/v1",
    event_type="UserPromptSubmit",
)
print(f"Done in {time.monotonic()-t0:.1f}s")

out = "eval/corpora/enriched_workflow_v3"
Path(out).mkdir(parents=True, exist_ok=True)
save_rules_json(workflow_expanded, out)
total = sum(len(r.expansions) for r in workflow_expanded)
print(f"Workflow v3: {len(workflow_expanded)} rules, {total} expansions (avg {total/len(workflow_expanded):.1f})")

# Quick quality check
print("\n=== Sample expansions (basic) ===")
for r in basic_expanded[:3]:
    print(f"\nRule: {r.text[:60]}...")
    for e in r.expansions[:5]:
        print(f"  - {e}")

print("\n=== Sample expansions (workflow) ===")
for r in workflow_expanded[:3]:
    print(f"\nRule: {r.text[:60]}...")
    for e in r.expansions[:5]:
        print(f"  - {e}")

# Dedup stats
print("\n=== Dedup stats ===")
for label, rules in [("basic", basic_expanded), ("workflow", workflow_expanded)]:
    counts = [len(r.expansions) for r in rules]
    print(f"{label}: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}, total={sum(counts)}")

# Compare with v1 if available
for label, v1_path in [("basic", "eval/corpora/enriched_basic"), ("workflow", "eval/corpora/enriched_workflow")]:
    rj = Path(v1_path) / "rules.json"
    if rj.exists():
        import json
        v1 = json.loads(rj.read_text())
        v1_total = sum(len(r.get("expansions", [])) for r in v1)
        v1_counts = [len(r.get("expansions", [])) for r in v1]
        print(f"\n{label} v1: {len(v1)} rules, {v1_total} expansions (avg {v1_total/len(v1):.1f}, min={min(v1_counts)}, max={max(v1_counts)})")
