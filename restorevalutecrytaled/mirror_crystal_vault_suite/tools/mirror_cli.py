#!/usr/bin/env python3
"""MIRROR 7 // Interactive Local CLI & Testing Console

Allows testing Mirror 7 locally from the terminal:
  - Interactive Conversational REPL with Memory Graph retention
  - Direct Symbolic Cerebrum Math & Reasoning Engine
  - Non-Vacuous TDD Synthesis & Sandbox Verification
  - Real-time Self-Teaching (/teach)
  - Memory Management (/memory, /remember, /forget, /undo)
  - One-shot execution: python mirror_cli.py "What color is the sky?"
"""

import sys
import os
import time
import argparse
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Locate the repository root that owns mirror7_backend.
from pathlib import Path

REPO_ROOT = next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / "mirror7_backend" / "service.py").is_file()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("mirror7_backend was not found above this restored bundle")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mirror7_backend.service import BackendService


BANNER = r"""
========================================================================
  __  __ _____ _____  _____   ____  _____     ______
 |  \/  |_   _|  __ \|  __ \ / __ \|  __ \   |____  |
 | \  / | | | | |__) | |__) | |  | | |__) |      / / 
 | |\/| | | | |  _  /|  _  /| |  | |  _  /      / /  
 | |  | |_| |_| | \ \| | \ \| |__| | | \ \     / /   
 |_|  |_|_____|_|  \_\_|  \_\\____/|_|  \_\   /_/    
                                                     
 Mirror 7 // CPU-first local research engine
 Uses the repository backend and its configured vault path
========================================================================
Type your prompt directly or use commands:
  /help       - Show available commands & examples
  /math <exp> - Explicit Symbolic Cerebrum Math Solver
  /code <req> - TDD Code Synthesis & Sandbox Repair
  /teach      - Teach Mirror new knowledge: /teach <question> | <answer>
  /memory     - View your active session memory graph
  /remember   - Store key-value: /remember my_database = PostgreSQL
  /status     - Show vault size, cache metrics & engine status
  /web        - Show instructions to start the Web Studio UI
  /exit       - Exit console
========================================================================
"""


def format_response(res: dict) -> str:
    lines = []
    ans = res.get("answer", "")
    mode = res.get("mode", "unknown")
    conf = res.get("confidence", 0.0)
    lat = res.get("latency_ms", 0.0)
    rid = res.get("matched_rule_id")
    epistemic = res.get("epistemic_tier", "")
    abstained = res.get("abstained", False)

    if abstained:
        lines.append(f"\n[MIRROR ABSTAINED] (Confidence: {conf:.2f} | Latency: {lat:.2f}ms)")
        lines.append(f"{ans}")
        lines.append("\nTip: Teach Mirror this answer using: /teach <question> | <answer>")
    else:
        lines.append(f"\n[MIRROR 7 RESPONSE] (Mode: {mode} | Conf: {conf*100:.1f}% | Latency: {lat:.2f}ms | Tier: {epistemic})")
        if rid:
            lines.append(f"[Matched Vault Rule #{rid}]")
        lines.append("-" * 72)
        lines.append(ans)
        lines.append("-" * 72)

    traces = res.get("reasoning_trace", [])
    if traces:
        lines.append(f"[Reasoning Steps: {len(traces)} step(s)]")
        for t in traces[:3]:
            lines.append(f"  * {t}")
    return "\n".join(lines)


def run_interactive(session_id: str = "cli_local"):
    print(BANNER)
    print("Initializing Mirror 7 Backend Engine...")
    t0 = time.perf_counter()
    svc = BackendService(enable_crystal_auto_load=True)
    init_ms = (time.perf_counter() - t0) * 1000
    st = svc.status()
    vault_rules = st.get("crystal", {}).get("rule_count", 0)
    print(f"Backend Ready in {init_ms:.1f}ms! Loaded {vault_rules:,} rules into DiskCrystalVault.")
    print(f"Active Session: '{session_id}' (Memory retention enabled)\n")

    while True:
        try:
            raw = input("Mirror-7> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Mirror 7. Goodbye!")
            break

        if not raw:
            continue

        low = raw.lower()
        if low in {"/exit", "/quit", "exit", "quit"}:
            print("Shutting down Mirror 7. Goodbye!")
            break

        if low in {"/help", "help", "?"}:
            print("\nCommands:")
            print("  <any prompt>                 - Normal query to Mirror (Grover Vault + Math Core)")
            print("  /math <expression/word-prob> - Route directly to Cerebrum symbolic math solver")
            print("  /code <prompt>               - Synthesize code with TDD verification")
            print("  /teach <Q> | <A>             - Crystallize new rule into vault")
            print("  /memory                      - List all remembered session entities")
            print("  /remember <key> = <val>      - Save custom key/val to Memory Graph")
            print("  /forget <key>                - Remove entity from Memory Graph")
            print("  /undo                        - Undo last memory mutation")
            print("  /status                      - Show engine health and cache statistics")
            print("  /downvote <id>               - Downvote and auto-retire false or broken rules")
            print("  /upvote <id>                 - Upvote verified rules")
            print("  /purge_domain <domain>       - Surgically delete all rules from a domain")
            print("  /compact                     - Run SQLite VACUUM and reclaim free disk space")
            print("  /gaps                        - View top unanswered topics logged from user queries")
            print("  /gap_prompt [topic]          - Generate ready-to-run ChatGPT prompt to fill missing knowledge")
            print("  /ingest <path> [domain]      - Zero-prompt ingestion of markdown/code file or directory")
            print("  /dream [min_confidence]      - Consolidate session memories into permanent crystal rules")
            print("  /web                         - Print instructions to launch the Web Studio")
            print()
            continue

        if low in {"/web", "web"}:
            print("\nTo launch the Quantum Mirror Web Studio Dashboard:")
            print("  1. In a separate terminal run:")
            print("     python -m mirror7_backend.http_server")
            print("  2. Open your web browser at:")
            print("     http://localhost:8787\n")
            continue

        if low in {"/status", "status"}:
            status = svc.status()
            cryst = status.get("crystal", {})
            print("\n=== MIRROR 7 ENGINE STATUS ===")
            print(f"Active Vault Rules: {cryst.get('rule_count', 0):,}")
            print(f"Vault Path:         {cryst.get('vault_path', 'N/A')}")
            cov = svc.crystal_coverage_map()
            print(f"Total Domains:      {cov.get('total_domains', 0)}")
            print(f"Abstention Rate:    {cov.get('abstention_rate', 0.0)*100:.1f}%")
            ents = svc.memory_list(session_id)
            print(f"Session Memory:     {len(ents)} active entities stored")
            print("==============================\n")
            continue

        if low.startswith("/math "):
            math_query = raw[6:].strip()
            t_start = time.perf_counter()
            sol = svc.cerebrum_solve(math_query)
            dur = (time.perf_counter() - t_start) * 1000
            print(f"\n[CEREBRUM MATH ENGINE] ({dur:.2f}ms)")
            print("-" * 72)
            print(sol.get("solution") or sol.get("answer", "No solution produced"))
            print("-" * 72)
            if sol.get("steps"):
                print("Steps:")
                for s in sol["steps"]:
                    print(f"  {s}")
            print()
            continue

        if low.startswith("/code "):
            code_query = raw[6:].strip()
            print("\nSynthesizing code in autonomous TDD sandbox...")
            t_start = time.perf_counter()
            res = svc.tdd_synthesize(code="", spec_desc=code_query)
            dur = (time.perf_counter() - t_start) * 1000
            print(f"\n[TDD ENGINE] Verified: {res.get('verified')} | Latency: {dur:.2f}ms")
            print("-" * 72)
            print(res.get("code", "No code produced"))
            print("-" * 72)
            if res.get("tests_executed"):
                print(f"Tests Passed: {res.get('tests_passed')}/{res.get('tests_executed')}")
            print()
            continue

        if low.startswith("/teach "):
            rest = raw[7:].strip()
            if "|" not in rest:
                print("Usage: /teach <question> | <answer>")
                continue
            q_part, a_part = rest.split("|", 1)
            taught = svc.crystal_teach(q_part.strip(), a_part.strip(), domain="local_cli_teach")
            print(f"[TEACH SUCCESS] Rule #{taught.get('rule_id')} crystallized into Vault! (Domain: {taught.get('domain')})\n")
            continue

        if low in {"/memory", "memory"}:
            ents = svc.memory_list(session_id)
            print(f"\n=== SESSION MEMORY GRAPH ('{session_id}') ===")
            if not ents:
                print("No entities stored yet. Store some using: /remember key = value")
            else:
                for e in ents:
                    print(f"  * {e.get('key')} = '{e.get('value')}' (type: {e.get('type')}, conf: {e.get('confidence')})")
            print("============================================\n")
            continue

        if low.startswith("/remember "):
            rest = raw[10:].strip()
            if "=" in rest:
                k, v = rest.split("=", 1)
            elif ":" in rest:
                k, v = rest.split(":", 1)
            else:
                k, v = rest, "true"
            ent = svc.memory_remember(session_id, k.strip(), v.strip())
            print(f"[MEMORY SAVED] Remembered '{k.strip()}' = '{v.strip()}'\n")
            continue

        if low.startswith("/forget "):
            k = raw[8:].strip()
            deleted = svc.memory_forget(session_id, k)
            if deleted:
                print(f"[MEMORY DELETED] Removed '{k}' from session memory.\n")
            else:
                print(f"Entity '{k}' not found in active session memory.\n")
            continue

        if low in {"/undo", "undo"}:
            undone = svc.memory_undo(session_id)
            if undone:
                print(f"[MEMORY UNDO] Successfully restored previous state for key: {undone.get('key')}\n")
            else:
                print("Undo stack is empty.\n")
            continue

        if low.startswith("/downvote "):
            try:
                rid = int(raw.split()[1])
                fb = svc.crystal_feedback(rid, is_positive=False)
                print(f"[FEEDBACK] Rule #{rid} downvoted! (pass: {fb.get('pass_count')}, fail: {fb.get('fail_count')}, active: {fb.get('active')})\n")
            except Exception as e:
                print(f"Error downvoting: {e}\n")
            continue

        if low.startswith("/upvote "):
            try:
                rid = int(raw.split()[1])
                fb = svc.crystal_feedback(rid, is_positive=True)
                print(f"[FEEDBACK] Rule #{rid} upvoted! (pass: {fb.get('pass_count')}, fail: {fb.get('fail_count')}, active: {fb.get('active')})\n")
            except Exception as e:
                print(f"Error upvoting: {e}\n")
            continue

        if low.startswith("/purge_domain "):
            dom = raw.split(None, 1)[1].strip()
            res = svc.crystal_purge_domain(dom)
            print(f"[PURGE] Domain '{dom}' purged! {res.get('purged_rules', 0)} rules removed from vault.\n")
            continue

        if low in {"/compact", "compact"}:
            print("Compacting vault on disk (VACUUM)...")
            res = svc.crystal_compact()
            print(f"[COMPACTION COMPLETE] {res.get('mb_before')} MB -> {res.get('mb_after')} MB (Reclaimed {res.get('bytes_reclaimed', 0):,} bytes)\n")
            continue

        if low in {"/gaps", "gaps"}:
            gaps = svc.crystal_get_gaps(limit=5)
            print("\n=== LOGGED COVERAGE GAPS (Missing Topics) ===")
            if not gaps:
                print("No coverage gaps logged yet! As users ask unanswered questions, they appear here.")
            else:
                for g in gaps:
                    print(f"  * Topic: '{g['topic']}' ({g['unanswered_count']} unanswered query occurrences)")
                    for q in g.get('sample_queries', [])[:2]:
                        print(f"      - {q}")
            print("=============================================\n")
            continue

        if low.startswith("/gap_prompt"):
            parts = raw.split(None, 1)
            t = parts[1].strip() if len(parts) > 1 else None
            prompt = svc.crystal_gap_prompt(topic=t)
            print("\n--- READY-TO-RUN CHATGPT PROMPT ---")
            print(prompt)
            print("-----------------------------------\n")
            continue

        if low.startswith("/ingest "):
            parts = raw.split(None, 2)
            p_arg = parts[1].strip()
            d_arg = parts[2].strip() if len(parts) > 2 else None
            res = svc.crystal_ingest_path(p_arg, domain=d_arg)
            if res.get("ok"):
                st = res.get("stats", {})
                print(f"[INGEST SUCCESS] Path '{p_arg}' ingested! Stats: {st}\n")
            else:
                print(f"[INGEST ERROR] {res.get('error')}\n")
            continue

        if low in {"/dream", "dream"} or low.startswith("/dream "):
            parts = raw.split()
            conf = float(parts[1]) if len(parts) > 1 else 0.85
            print(f"Consolidating active memories into Crystal Rules (min confidence: {conf})...")
            res = svc.memory_dream_consolidation(min_confidence=conf)
            if res.get("ok"):
                st = res.get("stats", {})
                print(f"[DREAM SUCCESS] Crystallized {st.get('rules_crystallized', 0)} new permanent rules from memories! (Scanned: {st.get('entities_scanned', 0)})\n")
            else:
                print(f"[DREAM ERROR] {res.get('error')}\n")
            continue

        # Standard Query
        res = svc.crystal_ask(raw, session_id=session_id)
        print(format_response(res))
        print()


def run_oneshot(query: str, session_id: str = "cli_oneshot"):
    svc = BackendService(enable_crystal_auto_load=True)
    res = svc.crystal_ask(query, session_id=session_id)
    print(format_response(res))


def main():
    parser = argparse.ArgumentParser(description="Mirror 7 Interactive CLI & Testing Tool")
    parser.add_argument("prompt", nargs="*", help="Optional query for one-shot execution")
    parser.add_argument("--session", default="cli_local", help="Session ID for cross-session memory")
    args = parser.parse_args()

    if args.prompt:
        run_oneshot(" ".join(args.prompt), session_id=args.session)
    else:
        run_interactive(session_id=args.session)


if __name__ == "__main__":
    main()
