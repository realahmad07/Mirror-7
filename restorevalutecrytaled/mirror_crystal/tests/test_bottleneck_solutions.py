"""
End-to-End Verification Suite for All 8 Bottleneck Solutions in Mirror Crystal.
"""
from __future__ import annotations

from pathlib import Path

from mirror_crystal import (
    CloudCrystalNode,
    DiskCrystalVault,
    MirrorCrystalEngine,
    PocketCrystalClient,
    SignedRuleCapsule,
)


def test_bottleneck_1_trust_tier_contradiction_supersession_and_poison_guard(tmp_path: Path):
    """Newer curated updates supersede stale facts, while Tier-3 untrusted lies cannot overwrite Tier-1/2 truths."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair("What is the latest stable Python release?", "Python 3.11 is the latest release.", trust_tier=2)
    assert "3.11" in eng.respond("What is the latest stable Python release?").answer

    # Curated 2026 update (equal trust_tier=2) cleanly supersedes stale 2023 fact!
    eng.ingest_pair("What is the latest stable Python release?", "Python 3.14 is the latest release.", trust_tier=2)
    assert "3.14" in eng.respond("What is the latest stable Python release?").answer

    # Untrusted troll (trust_tier=3) attempts to poison curated fact -> blocked!
    eng.ingest_pair("What is the latest stable Python release?", "Python 1.0 is the latest release.", trust_tier=3)
    assert "3.14" in eng.respond("What is the latest stable Python release?").answer

    # Same protection on DiskCrystalVault (.dcrystal)
    vault = DiskCrystalVault(tmp_path / "tier.dcrystal")
    vault.ingest_pair("Default gateway TLS version?", "TLS 1.2", trust_tier=2)
    vault.ingest_pair("Default gateway TLS version?", "TLS 1.3", trust_tier=2)
    vault.ingest_pair("Default gateway TLS version?", "SSL 2.0", trust_tier=3)
    assert "TLS 1.3" in vault.respond("Default gateway TLS version?").answer
    vault.close()


def test_bottleneck_2_micro_crystal_teleportation_offline_and_bounded_ram():
    """Weak-device PocketCrystalClient teleports 1.3 KB rules into <=4-rule RAM cap and solves unseen math OFFLINE."""
    cloud_eng = MirrorCrystalEngine()
    cloud_eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        "The exact modular result of 17^19 mod 1000000007 is 286765645.",
    )
    for i in range(10):
        cloud_eng.ingest_pair(f"Explain error code E{900 + i}", f"Error E{900 + i} means bus_timeout_{i}.")

    key = b"prod-signing-secret-key-2026"
    cloud = CloudCrystalNode(cloud_eng, signing_key=key, burst_capacity=50)
    pocket = PocketCrystalClient(cloud_node=cloud, verify_key=key, max_ram_rules=4)

    assert pocket.local_rule_count() == 0

    # 1st query misses Pocket RAM -> teleports generalized formula lock from Cloud!
    r_online = pocket.ask("Compute 23^29 mod 1000000007")
    assert str(pow(23, 29, 1000000007)) in r_online.answer
    assert pocket.stats["cloud_teleports"] == 1

    # Disconnect internet completely! Client solves brand-new unseen integers 100% OFFLINE in RAM!
    pocket.online = False
    r_offline = pocket.ask("Compute 83^97 mod 1000000007")
    assert str(pow(83, 97, 1000000007)) in r_offline.answer
    assert pocket.stats["local_ram_hits"] == 1

    # Reconnect and teleport 6 more rules -> LRU keeps Pocket RAM strictly <= 4 rules!
    pocket.online = True
    for i in range(6):
        pocket.ask(f"Explain error code E{900 + i}")
    assert pocket.local_rule_count() <= 4
    assert pocket.stats["evictions"] >= 3


def test_bottleneck_3_hmac_signed_capsule_tamper_defense_and_private_teach():
    """Tampered network capsules are blocked via constant-time HMAC, and user /teach never leaks to Cloud."""
    cloud_eng = MirrorCrystalEngine()
    cloud_eng.ingest_pair("Safe HMAC check in Python?", "Use hmac.compare_digest(a, b).")
    cloud = CloudCrystalNode(cloud_eng, signing_key=b"real-cloud-key")
    pocket = PocketCrystalClient(cloud_node=cloud, verify_key=b"real-cloud-key")

    # Attacker intercepts and modifies answer payload without signing key
    legit_cap = SignedRuleCapsule.sign(cloud_eng.rule_index.rules[0], b"real-cloud-key")
    forged_dict = dict(legit_cap.rule_dict)
    forged_dict["ar"] = "Use insecure a == b."
    accepted = pocket.ingest_signed_capsule({"rule_dict": forged_dict, "signature_hex": legit_cap.signature_hex})
    assert accepted is False
    assert pocket.stats["tampered_capsules_blocked"] == 1

    # User teaches private personal note -> stays in Pocket RAM, absent from Cloud!
    pocket.teach_private_local("My Wi-Fi router note?", "Living room shelf pin 4455.")
    assert "4455" in pocket.ask("My Wi-Fi router note?").answer
    assert cloud_eng.respond("My Wi-Fi router note?").mode == "unmatched_honest_abstain"


def test_bottleneck_4_on_device_privacy_scrubber_zero_cloud_leak():
    """Emails, API keys, and IPs are scrubbed on-device before cloud fallback and rehydrated locally."""
    cloud_eng = MirrorCrystalEngine()
    cloud_eng.ingest_pair(
        "How do I rotate leaked credential token on gateway host?",
        "Revoke the old token immediately in IAM, issue a scoped key, and reload gateway TLS config.",
    )
    cloud = CloudCrystalNode(cloud_eng)
    pocket = PocketCrystalClient(cloud_node=cloud)

    sensitive_prompt = (
        "How do I rotate leaked credential token sk-live-SUPERSECRET998877 for admin@mirror-ai.org on gateway host 192.168.42.9?"
    )
    res = pocket.ask(sensitive_prompt)
    assert "Revoke the old token immediately" in res.answer

    # Audit Cloud Server logs: ZERO raw secrets ever left the user's device!
    assert len(cloud.audit_received_queries) == 1
    logged = cloud.audit_received_queries[0]
    assert "sk-live-SUPERSECRET998877" not in logged
    assert "admin@mirror-ai.org" not in logged
    assert "192.168.42.9" not in logged


def test_bottleneck_5_compound_multi_intent_prompt_splitting():
    """Multi-part human questions split cleanly and stitch verified answers from different subsystems."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        "The exact modular result of 17^19 mod 1000000007 is 286765645.",
    )
    eng.observe_unlabeled_text("ceramic_mug is a fragile_item.\nif fragile_item, dropping_shatters_it.")

    res = eng.respond("Compute 29^31 mod 1000000007 and also what happens to a ceramic_mug?")
    assert res.mode == "compound_multi_intent"
    assert str(pow(29, 31, 1000000007)) in res.answer
    assert "dropping_shatters_it" in res.answer


def test_bottleneck_6_long_crash_log_sliding_window_anti_saturation():
    """200-line noisy server logs do not saturate 2,048-bit barcodes; diagnostic error window wins cleanly."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "FATAL Exception ZeroDivisionError in normalize_wavefunction",
        "Guard denominator with max(1e-12, norm) before dividing amplitudes in normalize_wavefunction.",
    )

    noisy_log = "\n".join(
        [f"2026-11-08 12:00:{i % 60:02d} INFO worker_{i} heartbeat nominal" for i in range(120)]
        + ["2026-11-08 12:02:01 FATAL Exception ZeroDivisionError in normalize_wavefunction at step 44"]
        + [f"2026-11-08 12:02:{i % 60:02d} DEBUG flushing metrics buffer {i}" for i in range(80)]
    )
    res = eng.respond(noisy_log)
    assert "max(1e-12, norm)" in res.answer


def test_bottleneck_7_negation_polarity_disambiguation():
    """Affirmative and negated security rules never swap answers despite sharing 90% of words."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "When should we enable gateway_debug_mode?",
        "Enable gateway_debug_mode ONLY inside isolated local staging containers.",
    )
    eng.ingest_pair(
        "When should we NEVER enable gateway_debug_mode?",
        "NEVER enable gateway_debug_mode on public production edge nodes.",
    )

    r_pos = eng.respond("When should we enable gateway_debug_mode?")
    r_neg = eng.respond("When should we NEVER enable gateway_debug_mode?")
    assert "isolated local staging" in r_pos.answer
    assert "public production edge" in r_neg.answer


def test_bottleneck_8_viral_cdn_edge_cache_and_anti_scraper_limiter():
    """CloudCrystalNode serves viral duplicate traffic from O(1) CDN cache and blocks scraper bots."""
    cloud_eng = MirrorCrystalEngine()
    cloud_eng.ingest_pair("What is a qubit?", "A qubit is a coherent two-state quantum system.")
    cloud = CloudCrystalNode(cloud_eng, teleports_per_sec=0.01, burst_capacity=3.0)

    # First viral query populates deterministic CDN cache; second hits CDN in O(1) RAM!
    p1 = cloud.handle_teleport_query("What is a qubit?", device_id="user_1")
    p2 = cloud.handle_teleport_query("What is a qubit?", device_id="user_2")
    assert p1["cdn_cached"] is False
    assert p2["cdn_cached"] is True
    assert cloud.cdn_hits == 1

    # Scraper bot fires rapid requests from single device -> rate-limited after burst=3!
    scraper = PocketCrystalClient(cloud_node=cloud, device_id="bot_scraper_99")
    statuses = [scraper.ask(f"Unknown scrape probe {i}").mode for i in range(5)]
    assert "cloud_rate_limited" in statuses
