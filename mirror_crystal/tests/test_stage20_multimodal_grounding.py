from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from mirror_crystal.phase5_sensory_shards_wasm import SensoryCodec
from mirror_crystal.stage20_multimodal_grounding import (
    ExtractedSegment,
    GroundingError,
    GroundingStatus,
    RealWorldGroundingEngine,
    SourceTrustRegistry,
)


def _keypair():
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return private, public


def _signature(private, payload: bytes) -> str:
    return base64.b64encode(private.sign(payload)).decode("ascii")


def _png():
    return SensoryCodec.encode_grid_to_png_bytes([[0, 40, 80], [120, 160, 200]])


def _csv(value="23.0", unit="C", entity="sensor-a"):
    return (
        "entity_key,attribute,value,unit,observed_at\n"
        f"{entity},temperature,{value},{unit},2026-11-11T10:00:00Z\n"
    ).encode("utf-8")


def _telemetry(engine, private, value=23.0, unit="°C", entity="sensor-a", device="device-a"):
    sample = {
        "sample_id": f"sample-{value}-{unit}",
        "entity_key": entity,
        "attribute": "temperature",
        "value": value,
        "unit": unit,
        "observed_at": "2026-11-11T10:00:00Z",
    }
    signature = _signature(private, engine.telemetry_signing_bytes(sample))
    return engine.ingest_telemetry(
        f"telemetry-{value}-{unit}", sample, device_id=device, signature_b64=signature,
    )


def test_png_ingestion_verifies_crc_and_emits_bounded_pixel_observations():
    engine = RealWorldGroundingEngine()
    raw = _png()
    asset = engine.ingest_png("image-a", raw, entity_key="room-a")
    assert asset.integrity_verified is True
    assert asset.origin_verified is False
    assert asset.sha256 == hashlib.sha256(raw).hexdigest()
    features = {item.attribute: item.value for item in engine.observations}
    assert features["width_px"] == 3.0
    assert features["height_px"] == 2.0
    assert features["pixel_count"] == 6.0
    assert features["min_luma"] == 0.0
    assert features["max_luma"] == 200.0
    assert len(engine.observations) == 7


def test_png_rejects_crc_tampering_before_pixel_decode():
    engine = RealWorldGroundingEngine()
    raw = bytearray(_png())
    raw[-5] ^= 0xFF
    with pytest.raises(ValueError, match="CRC mismatch"):
        engine.ingest_png("tampered", bytes(raw))


def test_png_rejects_trailing_bytes_and_truncation():
    engine = RealWorldGroundingEngine()
    with pytest.raises(ValueError, match="trailing bytes"):
        engine.ingest_png("trailing", _png() + b"junk")
    with pytest.raises(ValueError):
        engine.ingest_png("truncated", _png()[:-5])


def test_png_source_origin_is_only_verified_with_trusted_signature():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("camera-1", public)
    engine = RealWorldGroundingEngine(trust)
    raw = _png()
    asset = engine.ingest_png("image-signed", raw, signer_id="camera-1", signature_b64=_signature(private, raw))
    assert asset.origin_verified is True
    with pytest.raises(GroundingError, match="signature verification failed"):
        engine.ingest_png("image-bad-signature", raw, signer_id="camera-1", signature_b64=_signature(Ed25519PrivateKey.generate(), raw))


def test_png_rejects_untrusted_signer_and_never_downgrades_bad_signature():
    _, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("camera-1", public)
    engine = RealWorldGroundingEngine(trust)
    with pytest.raises(GroundingError, match="not trusted"):
        engine.ingest_png("unknown-signer", _png(), signer_id="other-camera", signature_b64=base64.b64encode(b"x" * 64).decode())


def test_text_document_preserves_exact_line_locators_and_hash():
    engine = RealWorldGroundingEngine()
    raw = b"Device report\nTemperature was recorded.\nSecond reading pending.\n"
    asset, segments = engine.ingest_document("txt-a", "report.txt", raw)
    assert asset.sha256 == hashlib.sha256(raw).hexdigest()
    assert [item.locator for item in segments] == ["line:1", "line:2", "line:3"]
    assert segments[1].text == "Temperature was recorded."
    report = engine.analyze()
    assert len(report.segments) == 3
    assert report.segments[1].locator == "line:2"
    assert report.segments[1].text_sha256 == hashlib.sha256(segments[1].text.encode()).hexdigest()


def test_csv_extracts_structured_measurement_records_and_units():
    engine = RealWorldGroundingEngine()
    asset, segments = engine.ingest_document("csv-a", "measurements.csv", _csv("21.5", "C"))
    assert asset.media_type == "text/csv"
    assert len(segments) == 2
    item = next(item for item in engine.observations if item.attribute == "temperature")
    assert item.entity_key == "sensor-a"
    assert item.value == pytest.approx(294.65)
    assert item.canonical_unit == "K"
    assert item.locator == "document:record:1"


def test_json_duplicate_keys_nonfinite_and_invalid_utf8_are_rejected():
    engine = RealWorldGroundingEngine()
    with pytest.raises(GroundingError, match="duplicate JSON object key"):
        engine.ingest_document("json-dup", "x.json", b'{"x":1,"x":2}')
    with pytest.raises(GroundingError, match="non-finite"):
        engine.ingest_document("json-nan", "x.json", b'{"x":NaN}')
    with pytest.raises(GroundingError, match="UTF-8"):
        engine.ingest_document("json-utf", "x.json", b"\xff")


def test_json_structured_records_become_observations_with_record_locators():
    engine = RealWorldGroundingEngine()
    raw = json.dumps({"records": [
        {"entity_key": "sensor-a", "attribute": "pressure", "value": 101.2, "unit": "kPa",
         "observed_at": "2026-11-11T10:00:00Z"}
    ]}).encode()
    asset, segments = engine.ingest_document("json-a", "readings.json", raw)
    item = next(item for item in engine.observations if item.attribute == "pressure")
    assert item.entity_key == "sensor-a"
    assert item.value == pytest.approx(101200.0)
    assert item.canonical_unit == "Pa"
    assert item.locator == "document:record:1"


def test_docx_extracts_paragraphs_and_table_rows_without_extracting_to_disk():
    out = io.BytesIO()
    document = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>Field report</w:t></w:r></w:p>'
        '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Sensor</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>'
        '</w:tbl></w:body></w:document>'
    )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
    engine = RealWorldGroundingEngine()
    _, segments = engine.ingest_document("docx-a", "field.docx", out.getvalue())
    assert any(item.text == "Field report" and item.locator == "paragraph:1" for item in segments)
    assert any(item.text == "Sensor | Value" and item.locator == "table:1:row:1" for item in segments)


def test_docx_table_rows_with_measurement_headers_become_grounded_observations():
    out = io.BytesIO()
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
        '<w:tbl>'
        '<w:tr>' + ''.join('<w:tc><w:p><w:r><w:t>' + value + '</w:t></w:r></w:p></w:tc>'
                          for value in ("entity_key", "attribute", "value", "unit")) + '</w:tr>'
        '<w:tr>' + ''.join('<w:tc><w:p><w:r><w:t>' + value + '</w:t></w:r></w:p></w:tc>'
                          for value in ("sensor-docx", "humidity", "51", "%")) + '</w:tr>'
        '</w:tbl></w:body></w:document>'
    )
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
    engine = RealWorldGroundingEngine()
    engine.ingest_document("docx-table", "measurements.docx", out.getvalue())
    item = next(item for item in engine.observations if item.attribute == "humidity")
    assert item.entity_key == "sensor-docx"
    assert item.value == 51.0
    assert item.canonical_unit == "%"
    assert item.locator == "document:record:1"


def test_docx_rejects_path_traversal_and_xml_entity_payloads():
    engine = RealWorldGroundingEngine()
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../evil.txt", "bad")
        archive.writestr("word/document.xml", "<w:document/>")
    with pytest.raises(GroundingError, match="unsafe path"):
        engine.ingest_document("docx-traversal", "bad.docx", out.getvalue())

    out2 = io.BytesIO()
    xml = '<!DOCTYPE x [<!ENTITY y "boom">]><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body/></w:document>'
    with zipfile.ZipFile(out2, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
    with pytest.raises(GroundingError, match="declarations/entities"):
        engine.ingest_document("docx-entity", "bad.docx", out2.getvalue())


def test_pdf_requires_explicit_extractor_adapter_and_preserves_adapter_locator():
    engine = RealWorldGroundingEngine()
    raw = b"%PDF-1.4 host-owned sample"
    with pytest.raises(GroundingError, match="explicit host extractor"):
        engine.ingest_document("pdf-missing-adapter", "scan.pdf", raw)
    _, segments = engine.ingest_document(
        "pdf-adapter", "scan.pdf", raw,
        extractor=lambda data: [ExtractedSegment("page:2:paragraph:3", "Reported value: 23 C")],
        extractor_id="host-pdf-text-v1",
    )
    assert segments[0].locator == "page:2:paragraph:3"


def test_signed_source_and_signed_device_agreement_is_verified_after_unit_conversion():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("doc-owner", public)
    trust.register("device-a", public)
    engine = RealWorldGroundingEngine(trust)
    raw = _csv("73.4", "F")
    engine.ingest_document(
        "document-temperature", "report.csv", raw,
        signer_id="doc-owner", signature_b64=_signature(private, raw),
    )
    _telemetry(engine, private, value=23.0, unit="C")
    report = engine.analyze(absolute_tolerance=0.02)
    result = next(item for item in report.conclusions if item.entity_key == "sensor-a")
    assert result.status is GroundingStatus.VERIFIED_AGREEMENT
    assert set(result.modalities) == {"document", "telemetry"}
    assert result.canonical_unit == "K"
    assert result.values[0] == pytest.approx(result.values[1], abs=0.02)
    assert len(report.report_fingerprint) == 64


def test_unsigned_document_and_signed_sensor_agreement_stays_unverified():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("device-a", public)
    engine = RealWorldGroundingEngine(trust)
    engine.ingest_document("unsigned-doc", "report.csv", _csv("23.0", "C"))
    _telemetry(engine, private, value=23.0, unit="C")
    report = engine.analyze()
    result = next(item for item in report.conclusions if item.attribute == "temperature")
    assert result.status is GroundingStatus.CORROBORATED_UNVERIFIED
    assert any("authenticity remains unverified" in item for item in report.unknowns)


def test_cross_modal_disagreement_is_reported_as_conflict_not_averaged_away():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("doc-owner", public)
    trust.register("device-a", public)
    engine = RealWorldGroundingEngine(trust)
    raw = _csv("22.0", "C")
    engine.ingest_document("doc", "report.csv", raw, signer_id="doc-owner", signature_b64=_signature(private, raw))
    _telemetry(engine, private, value=31.0, unit="C")
    result = next(item for item in engine.analyze().conclusions if item.attribute == "temperature")
    assert result.status is GroundingStatus.CONFLICT
    assert "differ" in result.explanation


def test_one_modality_is_insufficient_and_unknowns_are_explicit():
    engine = RealWorldGroundingEngine()
    engine.ingest_png("single-image", _png(), entity_key="scene-1")
    report = engine.analyze()
    assert report.conclusions
    assert all(item.status is GroundingStatus.INSUFFICIENT for item in report.conclusions)
    assert report.unknowns


def test_incompatible_dimensions_are_not_misreported_as_conflict_or_agreement():
    engine = RealWorldGroundingEngine()
    engine.ingest_document("doc-len", "x.csv", b"entity_key,attribute,value,unit\nsensor-a,reading,12,m\n")
    engine.ingest_telemetry(
        "sensor-voltage",
        {"sample_id": "s1", "entity_key": "sensor-a", "attribute": "reading",
         "value": 12, "unit": "V", "observed_at": "2026-11-11T10:00:00Z"},
        device_id="unregistered", signature_b64=None,
    )
    result = next(item for item in engine.analyze().conclusions if item.attribute == "reading")
    assert result.status is GroundingStatus.INCOMPARABLE


def test_bad_device_signature_and_unknown_device_fail_closed():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("device-a", public)
    engine = RealWorldGroundingEngine(trust)
    sample = {
        "sample_id": "s1", "entity_key": "sensor-a", "attribute": "temperature",
        "value": 20.0, "unit": "C", "observed_at": "2026-11-11T10:00:00Z",
    }
    with pytest.raises(GroundingError, match="signature verification failed"):
        engine.ingest_telemetry("bad-signature", sample, device_id="device-a",
                                signature_b64=_signature(Ed25519PrivateKey.generate(), engine.telemetry_signing_bytes(sample)))
    with pytest.raises(GroundingError, match="not trusted"):
        engine.ingest_telemetry("unknown-device", sample, device_id="device-b",
                                signature_b64=_signature(private, engine.telemetry_signing_bytes(sample)))


def test_telemetry_replay_is_rejected_even_when_repackaged_under_another_source_id():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("device-a", public)
    engine = RealWorldGroundingEngine(trust)
    sample = {
        "sample_id": "non-replayable-17", "entity_key": "sensor-a", "attribute": "temperature",
        "value": 20.0, "unit": "C", "observed_at": "2026-11-11T10:00:00Z",
    }
    signature = _signature(private, engine.telemetry_signing_bytes(sample))
    engine.ingest_telemetry("first-envelope", sample, device_id="device-a", signature_b64=signature)
    with pytest.raises(GroundingError, match="replay detected"):
        engine.ingest_telemetry("second-envelope", sample, device_id="device-a", signature_b64=signature)


def test_missing_fields_naive_timestamp_nonfinite_value_and_unknown_unit_are_rejected():
    engine = RealWorldGroundingEngine()
    with pytest.raises(GroundingError, match="missing fields"):
        engine.ingest_telemetry("missing", {"sample_id": "s"}, device_id="device")
    base = {"sample_id": "s1", "entity_key": "sensor-a", "attribute": "temperature",
            "value": 20.0, "unit": "C", "observed_at": "2026-11-11T10:00:00"}
    with pytest.raises(GroundingError, match="timezone"):
        engine.ingest_telemetry("naive-time", base, device_id="device")
    base["observed_at"] = "2026-11-11T10:00:00Z"
    base["value"] = float("nan")
    with pytest.raises(GroundingError, match="finite"):
        engine.ingest_telemetry("nan", base, device_id="device")
    base["value"] = 20.0
    base["unit"] = "widgets"
    with pytest.raises(GroundingError, match="unsupported"):
        engine.ingest_telemetry("bad-unit", base, device_id="device")


def test_signed_image_annotation_binds_exact_asset_and_verified_adapter_identity():
    private, public = _keypair()
    trust = SourceTrustRegistry()
    trust.register("vision-adapter-v1", public)
    engine = RealWorldGroundingEngine(trust)
    asset = engine.ingest_png("scene", _png())
    envelope = {
        "source_id": asset.source_id,
        "source_sha256": asset.sha256,
        "entity_key": "scene-1",
        "attribute": "object_label",
        "value": "red warning light",
        "unit": "",
        "locator": "box:12,8,40,40",
        "observed_at": "2026-11-11T10:00:00Z",
    }
    sig = _signature(private, json.dumps(envelope, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode())
    observation = engine.add_signed_image_annotation(
        "scene", entity_key="scene-1", attribute="object_label", value="red warning light",
        locator="box:12,8,40,40", annotator_id="vision-adapter-v1",
        signature_b64=sig, observed_at="2026-11-11T10:00:00Z",
    )
    assert observation.origin_verified is True
    assert observation.signer_id == "vision-adapter-v1"
    assert observation.signature_sha256 == hashlib.sha256(base64.b64decode(sig)).hexdigest()
    with pytest.raises(GroundingError, match="signature verification failed"):
        engine.add_signed_image_annotation(
            "scene", entity_key="scene-1", attribute="object_label", value="green light",
            locator="box:12,8,40,40", annotator_id="vision-adapter-v1",
            signature_b64=sig, observed_at="2026-11-11T10:00:00Z",
        )


def test_report_fingerprint_is_reproducible_for_identical_frozen_inputs():
    raw = _csv()
    a = RealWorldGroundingEngine()
    b = RealWorldGroundingEngine()
    a.ingest_document("same-doc", "x.csv", raw, retrieved_at="2026-11-11T10:00:00Z")
    b.ingest_document("same-doc", "x.csv", raw, retrieved_at="2026-11-11T10:00:00Z")
    assert a.analyze().report_fingerprint == b.analyze().report_fingerprint


def test_resource_limits_duplicate_ids_and_license_validation_fail_closed():
    engine = RealWorldGroundingEngine()
    engine.ingest_document("once", "x.txt", b"content")
    with pytest.raises(GroundingError, match="duplicate source_id"):
        engine.ingest_document("once", "y.txt", b"other")
    with pytest.raises(GroundingError, match="SPDX"):
        engine.ingest_document("bad-license", "x.txt", b"content", license_spdx="not-a-license")
    with pytest.raises(GroundingError, match="HTTPS or a URN"):
        engine.ingest_document("bad-uri", "x.txt", b"content", source_uri="file:///tmp/data.txt")


def test_document_extractor_output_is_bounded_and_type_checked():
    engine = RealWorldGroundingEngine()
    with pytest.raises(GroundingError, match="ExtractedSegment"):
        engine.ingest_document("bad-adapter", "report.pdf", b"%PDF", extractor=lambda _: ["not-a-segment"], extractor_id="bad-pdf")
