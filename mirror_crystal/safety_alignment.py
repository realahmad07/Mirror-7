"""Stage 13: Safety, Privacy & Alignment Subsystem.

Provides comprehensive alignment, defensive security, and privacy protection:
1. Safe Refusal Contracts: High-severity threat detection (malware, exploits, wiretapping,
   credential theft) paired with technical defensive engineering analysis.
2. Prompt Injection Defense: Detection and neutralization of direct/indirect injection,
   delimiter escaping, persona hijacking (DAN/Developer mode), and taint tracking.
3. Privacy & Credential Protection: System file shielding (/etc/shadow, SSH keys),
   Luhn-verified credit card redactor, SSN/email scrubbing, and audit trails.
4. Execution Safety Guard: Deep AST exploit prevention (dunder traversal, monkey-patching,
   resource explosion prevention) building upon ZeroTrustSandboxJudge.
5. Central SafetyAlignmentManager: Unified pipeline orchestrating evaluation, sanitization,
   Prometheus metrics, and audit logging.
"""
from __future__ import annotations

import ast
import base64
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .production_readiness import MetricsCollector, StructuredLogger


# ============================================================================
# 1. Safety Categories, Enums, and Data Models
# ============================================================================

class SafetyCategory(str, Enum):
    """Taxonomy of safety and alignment classifications."""
    SAFE = "SAFE"
    MALWARE_EXPLOIT = "MALWARE_EXPLOIT"
    SYSTEM_INTRUSION = "SYSTEM_INTRUSION"
    CREDENTIAL_THEFT = "CREDENTIAL_THEFT"
    SURVEILLANCE_WIRETAPPING = "SURVEILLANCE_WIRETAPPING"
    CBRN_HAZARDS = "CBRN_HAZARDS"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    PII_LEAKAGE = "PII_LEAKAGE"
    UNSAFE_CODE_EXECUTION = "UNSAFE_CODE_EXECUTION"


@dataclass(frozen=True)
class RefusalContract:
    """Deterministic refusal message paired with constructive defensive analysis."""
    policy_refusal: str
    defensive_analysis: str
    category: SafetyCategory
    remediation_advice: List[str] = field(default_factory=list)

    def format_full_response(self) -> str:
        """Renders the combined refusal and defensive engineering analysis."""
        lines = [
            f"### Security & Alignment Policy Refusal [{self.category.value}]",
            "",
            self.policy_refusal,
            "",
            "**Defensive Engineering & Threat Mitigation Analysis:**",
            self.defensive_analysis,
        ]
        if self.remediation_advice:
            lines.append("")
            lines.append("**Industry Standard Best Practices:**")
            for item in self.remediation_advice:
                lines.append(f"- {item}")
        return "\n".join(lines)


@dataclass
class SafetyEvaluationResult:
    """Outcome of safety, alignment, and privacy inspection."""
    is_safe: bool
    category: SafetyCategory
    risk_score: float  # 0.0 (benign) to 1.0 (critical threat)
    refusal_contract: Optional[RefusalContract] = None
    sanitized_text: str = ""
    detected_violations: List[str] = field(default_factory=list)
    tainted: bool = False
    latency_ms: float = 0.0


# ============================================================================
# 2. Safe Refusal Engine & Threat Modeling
# ============================================================================

class SafeRefusalEngine:
    """Evaluates high-severity safety violations and produces defensive analysis."""

    def __init__(self):
        # High-threat patterns mapping to category and policy
        self._malware_patterns = [
            re.compile(r"\b(polymorphic\s+ransomware|create\s+ransomware|ransomware\s+payload)\b", re.I),
            re.compile(r"\b(evad(e|ing)\s+antivirus|antivirus\s+bypass|bypass\s+edr)\b", re.I),
            re.compile(r"\b(write\s+a\s+trojan|keylogger\s+payload|worm\s+propagation)\b", re.I),
        ]
        self._intrusion_patterns = [
            re.compile(r"\b(exploit\s+an?\s+unpatched|buffer\s+overflow\s+exploit|remote\s+code\s+execution\s+payload)\b", re.I),
            re.compile(r"\b(zero-day\s+exploit|kernel\s+privilege\s+escalation\s+script)\b", re.I),
            re.compile(r"\b(shellcode\s+injection|rop\s+chain\s+generator)\b", re.I),
        ]
        self._credential_patterns = [
            re.compile(r"\b(extract.*\/etc\/shadow|dump.*password\s+hashes|steal.*api\s+secrets)\b", re.I),
            re.compile(r"\b(mimikatz.*dump|dump\s+lsass|harvest\s+credentials)\b", re.I),
        ]
        self._wiretapping_patterns = [
            re.compile(r"\b(secretly\s+wiretap|covertly\s+eavesdrop|intercept\s+colleague.*messages)\b", re.I),
            re.compile(r"\b(unauthorized\s+surveillance|spyware\s+for\s+phone|covert\s+audio\s+recording\s+malware)\b", re.I),
        ]

    def evaluate_threat(self, text: str) -> Tuple[bool, SafetyCategory, float, Optional[RefusalContract]]:
        """Analyzes text for malicious intent and builds a refusal contract if needed."""
        # 1. Malware and Ransomware
        for p in self._malware_patterns:
            if p.search(text):
                contract = RefusalContract(
                    policy_refusal="I cannot assist with designing, generating, or distributing ransomware, malware, or antivirus evasion tools.",
                    defensive_analysis=(
                        "Modern Endpoint Detection & Response (EDR) systems rely on runtime behavioral heuristics, "
                        "kernel-level API monitoring, and memory entropy analysis rather than static signatures alone."
                    ),
                    category=SafetyCategory.MALWARE_EXPLOIT,
                    remediation_advice=[
                        "Implement immutable offline backups with append-only access controls.",
                        "Enforce strict Zero-Trust microsegmentation across production networks.",
                        "Deploy application allow-listing and behavioral EDR detection.",
                    ],
                )
                return False, SafetyCategory.MALWARE_EXPLOIT, 0.95, contract

        # 2. System Intrusion & Exploitation
        for p in self._intrusion_patterns:
            if p.search(text):
                contract = RefusalContract(
                    policy_refusal="I cannot provide functional exploits, shellcode, or scripts designed to attack computer systems.",
                    defensive_analysis=(
                        "Memory corruption vulnerabilities (such as buffer overflows) stem from missing boundary checks in low-level memory operations. "
                        "Operating systems deploy structural compiler mitigations to protect processes at runtime."
                    ),
                    category=SafetyCategory.SYSTEM_INTRUSION,
                    remediation_advice=[
                        "Address Space Layout Randomization (ASLR): Randomizes stack, heap, and library base addresses.",
                        "Stack Canaries: Inserts guard values before return addresses to detect stack frame corruption.",
                        "Data Execution Prevention (DEP/NX): Marks writable memory segments as strictly non-executable.",
                        "Adopt memory-safe systems programming languages (e.g. Rust, modern C++ with bounds checks).",
                    ],
                )
                return False, SafetyCategory.SYSTEM_INTRUSION, 0.98, contract

        # 3. Credential Theft & Shadow Dump
        for p in self._credential_patterns:
            if p.search(text):
                contract = RefusalContract(
                    policy_refusal="I cannot assist with unauthorized extraction, dumping, or harvesting of private credentials or authentication files.",
                    defensive_analysis=(
                        "Authentication stores such as /etc/shadow or credential vaults hold salted password digests. "
                        "Exposure compromises identity boundaries across the organization."
                    ),
                    category=SafetyCategory.CREDENTIAL_THEFT,
                    remediation_advice=[
                        "Store credentials exclusively in dedicated secret management engines (e.g. HashiCorp Vault, AWS Secrets Manager).",
                        "Enforce robust key-derivation functions (Argon2id, PBKDF2 with high iteration counts).",
                        "Apply Principle of Least Privilege: restrict administrative filesystem access using strict POSIX ACLs.",
                    ],
                )
                return False, SafetyCategory.CREDENTIAL_THEFT, 0.95, contract

        # 4. Surveillance & Wiretapping
        for p in self._wiretapping_patterns:
            if p.search(text):
                contract = RefusalContract(
                    policy_refusal="I cannot assist with unauthorized interception, wiretapping, or covert surveillance of communications.",
                    defensive_analysis=(
                        "End-to-End Encryption (E2EE) protocols (such as Signal's Double Ratchet) mathematically ensure that "
                        "only participating endpoints possess decryption keys, preserving privacy against network intermediaries."
                    ),
                    category=SafetyCategory.SURVEILLANCE_WIRETAPPING,
                    remediation_advice=[
                        "Double Ratchet Algorithm: Combines Diffie-Hellman ratcheting with symmetric KDF chains.",
                        "Forward Secrecy: Compromise of current keys does not expose past session communications.",
                        "Break-in Recovery: Future communications self-heal against temporary key compromise.",
                    ],
                )
                return False, SafetyCategory.SURVEILLANCE_WIRETAPPING, 0.90, contract

        return True, SafetyCategory.SAFE, 0.0, None


# ============================================================================
# 3. Prompt Injection & Adversarial Jailbreak Defense
# ============================================================================

class PromptInjectionGuard:
    """Detects and neutralizes prompt injections, jailbreaks, and delimiter tampering."""

    INJECTION_INDICATORS = [
        re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directives|rules)\b", re.I),
        re.compile(r"\b(you\s+are\s+now|act\s+as)\s+(DAN|unrestricted|jailbroken|godmode)\b", re.I),
        re.compile(r"\bdeveloper\s+mode\s+(enabled|activated|on)\b", re.I),
        re.compile(r"\bbypass\s+(all\s+)?(safety|ethical|content)\s+(filters|guidelines|rules)\b", re.I),
        re.compile(r"\bpretend\s+you\s+have\s+no\s+(rules|limits|guidelines)\b", re.I),
        re.compile(r"(```|---|<\|im_start\|>|<\|im_end\|>|\[SYSTEM\]|\[INST\])\s*system", re.I),
        re.compile(r"\bdisregard\s+system\s+prompt\b", re.I),
    ]

    def __init__(self):
        self._lock = threading.Lock()

    def check_injection(self, text: str) -> Tuple[bool, float, List[str]]:
        """Inspects text for prompt injection signatures and evasion techniques.

        Returns (is_clean, risk_score, detected_patterns).
        """
        detected: List[str] = []
        risk_score = 0.0

        # 1. Regex indicator detection
        for p in self.INJECTION_INDICATORS:
            match = p.search(text)
            if match:
                detected.append(f"indicator:{match.group(0)}")
                risk_score = max(risk_score, 0.85)

        # 2. Obfuscation detection: Base64 payload inspection
        b64_matches = re.findall(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}={0,2}(?![A-Za-z0-9+/])", text)
        for b64 in b64_matches:
            try:
                decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
                for p in self.INJECTION_INDICATORS:
                    if p.search(decoded):
                        detected.append(f"base64_obfuscated_injection:{b64[:16]}...")
                        risk_score = max(risk_score, 0.95)
            except Exception:
                pass

        # 3. Delimiter escape attempts
        escapes = len(re.findall(r"(?:```|<\|im_start\|>|\[\/INST\]|SYSTEM:|ADMIN:)", text, re.I))
        if escapes >= 3:
            detected.append(f"excessive_structural_delimiters:count={escapes}")
            risk_score = max(risk_score, 0.75)

        is_clean = len(detected) == 0
        return is_clean, round(risk_score, 3), detected

    def encapsulate_untrusted_input(self, untrusted_text: str, tag: str = "untrusted_input") -> str:
        """Enforces structural delimiter isolation preventing untrusted text from leaking as system instructions."""
        # Sanitize any delimiter collisions inside untrusted text
        safe_body = untrusted_text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")
        return f"<{tag} tainted=\"true\">\n{safe_body}\n</{tag}>"


# ============================================================================
# 4. Privacy, Credential & Data Protection
# ============================================================================

class PrivacyProtectionEngine:
    """Detects and redacts sensitive PII and confidential system credentials."""

    def __init__(self):
        # Social Security Numbers (US): XXX-XX-XXXX
        self._ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
        # Email addresses
        self._email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
        # System secrets and sensitive tokens
        self._token_patterns = [
            (re.compile(r"\beyJ[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b"), "[REDACTED_JWT]"),
            (re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.I), r"\1[REDACTED_BEARER_TOKEN]"),
            (re.compile(r"\b(AKIA[0-9A-Z]{16})\b"), "[REDACTED_AWS_KEY]"),
            (re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC )?PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
        ]

    @staticmethod
    def luhn_checksum_valid(card_number_str: str) -> bool:
        """Verifies credit card validity using the ISO/IEC 7812 Luhn algorithm."""
        digits = [int(c) for c in card_number_str if c.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return False
        total = 0
        reverse_digits = digits[::-1]
        for i, d in enumerate(reverse_digits):
            if i % 2 == 1:
                doubled = d * 2
                total += doubled - 9 if doubled > 9 else doubled
            else:
                total += d
        return (total % 10) == 0

    def scrub_pii_and_credentials(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Redacts PII and system secrets while preserving surrounding text context."""
        counts = {
            "ssn": 0,
            "credit_card": 0,
            "email": 0,
            "tokens": 0,
        }

        # 1. Scrub SSNs
        def _replace_ssn(m: re.Match) -> str:
            counts["ssn"] += 1
            return "[REDACTED_SSN]"

        sanitized = self._ssn_pattern.sub(_replace_ssn, text)

        # 2. Scrub Luhn-verified Credit Card Numbers
        # Candidate 13-19 digit strings with optional hyphens or spaces
        cc_candidates = re.finditer(r"\b(?:\d[ -]*?){13,19}\b", sanitized)
        # To avoid index drift, process in reverse
        for match in reversed(list(cc_candidates)):
            cand_str = match.group(0)
            cleaned_digits = re.sub(r"[^\d]", "", cand_str)
            if self.luhn_checksum_valid(cleaned_digits):
                counts["credit_card"] += 1
                sanitized = sanitized[:match.start()] + "[REDACTED_CREDIT_CARD]" + sanitized[match.end():]

        # 3. Scrub Emails
        def _replace_email(m: re.Match) -> str:
            counts["email"] += 1
            return "[REDACTED_EMAIL]"

        sanitized = self._email_pattern.sub(_replace_email, sanitized)

        # 4. Scrub Tokens & Keys
        for pattern, replacement in self._token_patterns:
            if pattern.search(sanitized):
                counts["tokens"] += len(pattern.findall(sanitized))
                sanitized = pattern.sub(replacement, sanitized)

        return sanitized, counts


# ============================================================================
# 5. Deep AST Execution Safety Guard
# ============================================================================

class ExecutionSafetyGuard:
    """Zero-Trust AST validation for generated code, preventing sandbox escapes."""

    FORBIDDEN_ATTRIBUTES: Set[str] = {
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__builtins__",
        "__import__",
        "gi_frame",
        "f_locals",
        "f_globals",
    }

    FORBIDDEN_CALLS: Set[str] = {
        "eval", "exec", "compile", "open", "input",
        "globals", "locals", "vars", "breakpoint", "exit", "quit",
        "os.system", "subprocess.Popen", "subprocess.run",
    }

    @classmethod
    def audit_python_code(cls, code: str) -> Tuple[bool, List[str]]:
        """Deeply inspects Python code AST for privilege escalation or sandbox escapes.

        Returns (is_safe, violation_reasons).
        """
        reasons: List[str] = []

        # 1. Base check via ZeroTrustSandboxJudge
        ok_base, base_reason = ZeroTrustSandboxJudge.is_ast_safe(code)
        if not ok_base:
            reasons.append(f"ZeroTrustSandboxJudge: {base_reason}")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, [f"Syntax error during AST parse: {e}"]

        # 2. Deep node inspection
        for node in ast.walk(tree):
            # Check attribute access for dunder escalation
            if isinstance(node, ast.Attribute):
                if node.attr in cls.FORBIDDEN_ATTRIBUTES:
                    reasons.append(f"Forbidden dunder attribute traversal: '{node.attr}'")

            # Check for direct calls to forbidden functions
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    val_id = getattr(node.func.value, "id", "")
                    func_name = f"{val_id}.{node.func.attr}" if val_id else node.func.attr

                if func_name in cls.FORBIDDEN_CALLS:
                    reasons.append(f"Forbidden execution call: '{func_name}'")

        is_safe = len(reasons) == 0
        return is_safe, reasons


# ============================================================================
# 6. Central Safety & Alignment Manager
# ============================================================================

class SafetyAlignmentManager:
    """Unified coordinator enforcing refusal contracts, injection defense, and privacy."""

    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        metrics: Optional[MetricsCollector] = None,
    ):
        self.logger = logger or StructuredLogger(name="mirror_safety")
        self.metrics = metrics or MetricsCollector()
        self.refusal_engine = SafeRefusalEngine()
        self.injection_guard = PromptInjectionGuard()
        self.privacy_engine = PrivacyProtectionEngine()
        self.execution_guard = ExecutionSafetyGuard()

    def evaluate_request(self, prompt: str, is_untrusted_source: bool = True) -> SafetyEvaluationResult:
        """Performs full multi-stage safety and alignment inspection on inbound request."""
        t0 = time.perf_counter()
        self.metrics.increment_counter("safety_evaluations_total")

        # 1. High-severity Threat & Refusal evaluation
        ok_threat, cat_threat, score_threat, contract = self.refusal_engine.evaluate_threat(prompt)
        if not ok_threat and contract:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.metrics.increment_counter("safety_refusals_total")
            self.logger.audit(
                action="SAFETY_POLICY_REFUSAL",
                actor="SafeRefusalEngine",
                status="BLOCKED",
                details={"category": cat_threat.value, "risk_score": score_threat},
            )
            return SafetyEvaluationResult(
                is_safe=False,
                category=cat_threat,
                risk_score=score_threat,
                refusal_contract=contract,
                sanitized_text=prompt,
                detected_violations=[f"Threat violation: {cat_threat.value}"],
                tainted=is_untrusted_source,
                latency_ms=elapsed,
            )

        # 2. Prompt Injection inspection
        ok_inj, score_inj, inj_violations = self.injection_guard.check_injection(prompt)
        if not ok_inj:
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.metrics.increment_counter("prompt_injections_blocked_total")
            self.logger.audit(
                action="PROMPT_INJECTION_BLOCKED",
                actor="PromptInjectionGuard",
                status="BLOCKED",
                details={"violations": inj_violations, "risk_score": score_inj},
            )
            injection_contract = RefusalContract(
                policy_refusal="I cannot process instructions containing prompt injections, role hijacking, or system bypass attempts.",
                defensive_analysis=(
                    "Prompt injection attacks attempt to manipulate LLM control flow by inserting adversarial delimiters "
                    "or overriding system persona anchors. Mirror 7 enforces strict delimiter isolation and input taint tracking."
                ),
                category=SafetyCategory.PROMPT_INJECTION,
                remediation_advice=[
                    "Employ structural data-plane and control-plane separation.",
                    "Tag untrusted inputs with taint metadata and sanitize before evaluation.",
                    "Disallow unescaped instruction delimiters in external user content.",
                ],
            )
            return SafetyEvaluationResult(
                is_safe=False,
                category=SafetyCategory.PROMPT_INJECTION,
                risk_score=score_inj,
                refusal_contract=injection_contract,
                sanitized_text=prompt,
                detected_violations=inj_violations,
                tainted=True,
                latency_ms=elapsed,
            )

        # 3. Privacy & PII scrubbing
        sanitized_text, pii_counts = self.privacy_engine.scrub_pii_and_credentials(prompt)
        total_pii = sum(pii_counts.values())
        if total_pii > 0:
            self.metrics.increment_counter("pii_redactions_total", total_pii)
            self.logger.info(f"Privacy engine redacted {total_pii} sensitive items", extra={"pii_counts": pii_counts})

        elapsed = (time.perf_counter() - t0) * 1000.0
        return SafetyEvaluationResult(
            is_safe=True,
            category=SafetyCategory.SAFE,
            risk_score=0.0,
            refusal_contract=None,
            sanitized_text=sanitized_text,
            detected_violations=[],
            tainted=is_untrusted_source,
            latency_ms=elapsed,
        )

    def guard_inference(
        self,
        prompt: str,
        inference_callable: Callable[[str], str],
    ) -> str:
        """Wraps inference execution with safety alignment evaluation and fallback refusal."""
        eval_result = self.evaluate_request(prompt)
        if not eval_result.is_safe and eval_result.refusal_contract:
            return eval_result.refusal_contract.format_full_response()

        # Execute on sanitized input
        return inference_callable(eval_result.sanitized_text)
