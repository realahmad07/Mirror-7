# Stage 20 — Multimodal & Real-World Grounding

## Goal and bounded scope

Stage 20 introduces a provenance-aware evidence path for images, documents and sensor telemetry. It separates four questions that must not be conflated:

1. Byte integrity: do the received bytes match the digest recorded by Mirror, and did supported file structures pass their integrity checks?
2. Origin authenticity: did a source identity already trusted by the host cryptographically sign this exact payload?
3. Observation extraction: which measurements, text passages or pixel-level features were deterministically derived, and where did they occur?
4. Cross-modal support: do independent modalities agree, disagree, or lack enough comparable evidence?

Implementation: mirror_crystal/stage20_multimodal_grounding.py.

## Image ingestion

RealWorldGroundingEngine.ingest_png accepts PNG bytes and enforces:
- PNG signature, chunk ordering, chunk bounds, CRC-32 for every chunk, required IHDR/IDAT/IEND, and no trailing bytes.
- 8-bit grayscale, RGB, grayscale-alpha and RGBA; indexed/palette, interlaced, unsupported bit-depth and unknown critical chunks fail closed.
- A 2048 × 2048 / 4,194,304-pixel limit, an 8 MiB compressed IDAT cap and an exact bounded zlib output length.
- A SHA-256 source record and deterministic pixel measurements: dimensions, pixel count, mean/minimum/maximum luma and luma variance.

The built-in image path is deliberately pixel-measurement extraction, not a general object-recognition or OCR model. A host perception system can add semantic observations through add_signed_image_annotation. Its Ed25519 signature binds the exact image hash, entity, attribute, value, unit, evidence locator and observation timestamp. The adapter's signer must be registered by the host.

## Document ingestion

Built-in parsers support:
- UTF-8 TXT and Markdown with line locators.
- CSV with row locators and structured measurement rows when columns include attribute and value (and optionally entity_key, unit, and observed_at).
- JSON with duplicate-key and non-finite-number rejection, preserving the full root payload and extracting structured records.
- DOCX paragraphs and table rows, without extracting archive entries to the filesystem. Entry count, decompressed size, compression ratio, traversal paths, duplicate entries, encrypted content, DTD and entity declarations are checked.

Every extracted document passage is retained as a DocumentEvidenceSegment containing its source digest, text digest, extractor identity, and locator such as line:12, csv:row:4, paragraph:3, table:1:row:2, or a host-supplied PDF page locator. PDF and other unsupported formats require an explicit host extractor that yields bounded ExtractedSegment values. Mirror does not silently pretend to OCR scanned pages.

## Telemetry and provenance

SourceTrustRegistry stores host-enrolled Ed25519 public keys. Signed image/document bytes are verified exactly as received. Telemetry signatures cover a canonical JSON record containing sample ID, entity key, attribute, value, unit and timezone-aware observation timestamp. Replayed (device ID, sample ID) pairs are rejected.

Supported unit conversions include Celsius/Fahrenheit/Kelvin, Pa/kPa/bar, m/cm/mm/km, kg/g, V/mV and percent. Unsupported units, non-finite values, booleans masquerading as numbers, missing metadata and naive timestamps are rejected. Unsigned telemetry can be retained as untrusted evidence, but it cannot be labelled origin-verified.

A trusted-key registry is a host security decision. Mirror verifies signatures against the keys it is given; it does not independently certify who owns a camera, document publisher, vision adapter or sensor.

## Cross-modal reasoning and epistemic outcomes

Observations are grouped only when they share a normalized entity key and attribute. Numeric values are converted to a canonical unit and compared pairwise using explicit absolute and relative tolerances. Categorical values are compared exactly after extraction. Mirror does not average away disagreement.

| Status | Meaning |
|---|---|
| verified_agreement | At least two distinct modalities and sources agree, and all observations have verified origin signatures. |
| corroborated_unverified | Independent modalities agree, but one or more origins are unsigned/untrusted. |
| conflict | Comparable observations disagree beyond the declared tolerance. |
| incomparable | Units or value dimensions do not match. |
| insufficient | Only one source modality is available; the report records the missing corroboration as an unknown. |

The report preserves assets, extracted passages, observations, source/signature digests, locators, conclusions, unknowns and a deterministic fingerprint. Integrity alone is never promoted to source authenticity; lexical or inferred agreement is never promoted to verified fact.

## Resource and adversarial controls

The engine caps source count, per-source bytes, total bytes, document segment count, extracted text, DOCX entries and observation count. It rejects duplicate source IDs, unsupported licenses/URIs, bad signatures, PNG CRC/decompression attacks, malformed archives/XML/JSON/CSV, path traversal, missing timestamps, replayed telemetry, unsupported units and conflicting measurements.

The acceptance suite includes positive controls, separate-source corroboration, signed and unsigned provenance paths, Fahrenheit/Celsius conversion, held-out modality combinations, explicit conflict and unknown cases, malformed/corrupted assets, signature spoofing, document archive attacks, and replay/metadata negative controls.

## Verification

Run:
- python -m pytest -q mirror_crystal/tests/test_stage20_multimodal_grounding.py
- python -m pytest -q mirror_crystal mirror_eval
- python -m compileall -q mirror_crystal

## Explicit limitations

- Native image analysis produces deterministic pixel statistics only. Semantic labels depend on a separately installed, host-trusted vision adapter.
- PDF/OCR extraction is a host adapter boundary; a scanned page without a trusted extractor remains unparsed.
- The engine does not retrieve files from the network, attest physical sensor calibration, prove a device's physical placement, or determine whether a host-enrolled public key belongs to the claimed organization.
- Numeric agreement is bounded evidence, not proof that sources describe the same physical event. Entity keys, timestamps, tolerances and trust enrollment must be chosen carefully by the host.
- Reports are in-memory objects. Durable evidence storage, key lifecycle, revocation and audit retention remain host responsibilities.

Stage 20 is therefore a bounded multimodal evidence and grounding layer, not a claim of unrestricted visual intelligence or autonomous real-world perception.
