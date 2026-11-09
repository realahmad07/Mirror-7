#!/usr/bin/env python3
"""Run real response probes against the Crystal brain inside crystal_4.zip.

From the Mirror-7 repository root:
  python scripts/test_crystal_archive_responses.py --brain datasets/crystal_archive/raw/crystal_4.zip

The report contains answer, mode, confidence, rule ID, epistemic tier and latency.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from mirror_crystal import MirrorCrystalEngine

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brain", type=Path, default=Path("datasets/crystal_archive/raw/crystal_4.zip"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.brain.is_file():
        parser.error(f"brain file does not exist: {args.brain}")
    engine = MirrorCrystalEngine.load(args.brain)
    probes = [
        {"name":"stored_rule_retrieval","prompt":"What is protocol code 1000?","expected_substring":"channel_0","purpose":"Retrieve a known record from the stored MCRYSTAL brain."},
        {"name":"unseen_exact_modular_arithmetic","prompt":"Compute 31^43 mod 1000000007","expected_substring":str(pow(31,43,1000000007)),"purpose":"Solve an unseen arithmetic input through Mirror's symbolic exact path."},
        {"name":"out_of_scope_abstention_probe","prompt":"Describe the governance history of the fictional city of Zorblax-999.","expected_substring":None,"purpose":"Observe whether the brain abstains on an unsupported fictional query."},
    ]
    results=[]; failures=[]
    for probe in probes:
        response=engine.respond(probe["prompt"],session_id="crystal-archive-response-test")
        passed=(probe["expected_substring"] is None or probe["expected_substring"] in response.answer)
        if not passed: failures.append(probe["name"])
        results.append({
          **probe,"answer":response.answer,"mode":response.mode,"confidence":response.confidence,
          "matched_rule_id":response.matched_rule_id,"epistemic_tier":response.epistemic_tier,
          "sandbox_verified":response.sandbox_verified,"latency_ms":response.latency_ms,
          "expected_check":"observational_only" if probe["expected_substring"] is None else ("PASS" if passed else "FAIL")
        })
    report={"brain":str(args.brain),"probes_total":len(results),"required_probe_failures":failures,"results":results,
      "note":"The unknown-domain probe is observational; a plausible non-abstaining answer requires review, not automatic acceptance."}
    rendered=json.dumps(report,indent=2,ensure_ascii=False); print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(rendered+"\n",encoding="utf-8")
    return 1 if failures else 0
if __name__=="__main__": raise SystemExit(main())
