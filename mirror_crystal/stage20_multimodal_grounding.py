"""Stage 20: bounded multimodal ingestion and evidence-grounded real-world reasoning.

The module deliberately separates byte integrity, source authenticity, extraction,
and cross-modal agreement. PNG analysis is deterministic and pixel-level; semantic
image annotations require a trusted host perception adapter signature. Text, CSV,
JSON and DOCX are parsed locally. PDF/other formats use an explicit host extractor.
No network retrieval, OCR claim, or unauthenticated sensor trust is implicit.
"""
from __future__ import annotations

import base64
import binascii
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import json
import math
import re
import struct
from typing import Any, Callable, Iterable, Mapping
import zipfile
import xml.etree.ElementTree as ET

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .phase5_sensory_shards_wasm import PNGSecurityError, SensoryCodec


class GroundingError(ValueError):
    """Invalid multimodal source, provenance, observation, or evidence contract."""


class Modality(str, Enum):
    IMAGE = "image"
    DOCUMENT = "document"
    TELEMETRY = "telemetry"


class GroundingStatus(str, Enum):
    VERIFIED_AGREEMENT = "verified_agreement"
    CORROBORATED_UNVERIFIED = "corroborated_unverified"
    CONFLICT = "conflict"
    INCOMPARABLE = "incomparable"
    INSUFFICIENT = "insufficient"


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_TIMESTAMP = re.compile(r".*(?:Z|[+-][0-9]{2}:[0-9]{2})\Z")
_UNIT_ALIASES = {
    "c": ("temperature", "K", 1.0, 273.15),
    "°c": ("temperature", "K", 1.0, 273.15),
    "degc": ("temperature", "K", 1.0, 273.15),
    "celsius": ("temperature", "K", 1.0, 273.15),
    "f": ("temperature", "K", 5.0 / 9.0, 255.3722222222222),
    "°f": ("temperature", "K", 5.0 / 9.0, 255.3722222222222),
    "degf": ("temperature", "K", 5.0 / 9.0, 255.3722222222222),
    "fahrenheit": ("temperature", "K", 5.0 / 9.0, 255.3722222222222),
    "k": ("temperature", "K", 1.0, 0.0),
    "kelvin": ("temperature", "K", 1.0, 0.0),
    "pa": ("pressure", "Pa", 1.0, 0.0),
    "kpa": ("pressure", "Pa", 1000.0, 0.0),
    "bar": ("pressure", "Pa", 100000.0, 0.0),
    "m": ("length", "m", 1.0, 0.0),
    "cm": ("length", "m", 0.01, 0.0),
    "mm": ("length", "m", 0.001, 0.0),
    "km": ("length", "m", 1000.0, 0.0),
    "kg": ("mass", "kg", 1.0, 0.0),
    "g": ("mass", "kg", 0.001, 0.0),
    "v": ("voltage", "V", 1.0, 0.0),
    "mv": ("voltage", "V", 0.001, 0.0),
    "%": ("ratio", "%", 1.0, 0.0),
    "percent": ("ratio", "%", 1.0, 0.0),
    "luma_0_255": ("image_luma", "luma_0_255", 1.0, 0.0),
    "px": ("pixel", "px", 1.0, 0.0),
    "": ("dimensionless", "", 1.0, 0.0),
}


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise GroundingError(f"value must be finite JSON data: {exc}") from exc


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _required_text(value: Any, field: str, limit: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GroundingError(f"{field} must be non-empty text")
    value = value.strip()
    if len(value) > limit:
        raise GroundingError(f"{field} exceeds {limit} characters")
    return value


def _utc_timestamp(value: str | None, field: str = "timestamp") -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise GroundingError(f"{field} must be ISO-8601 with an explicit timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GroundingError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GroundingError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _source_uri(value: str | None, digest: str) -> str:
    if value is None:
        return f"urn:mirror:asset:{digest}"
    uri = _required_text(value, "source_uri", 2048)
    if any(ord(ch) < 32 for ch in uri):
        raise GroundingError("source_uri contains control characters")
    if uri.startswith("https://") and len(uri) > 8:
        return uri
    if re.fullmatch(r"urn:[A-Za-z0-9][A-Za-z0-9-]{0,31}:.+", uri, re.IGNORECASE):
        return uri
    raise GroundingError("source_uri must be HTTPS or a URN; local file URLs are not accepted")


def _normalise_numeric(value: Any, unit: str) -> tuple[float, str, str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GroundingError("numeric observation value must be finite and must not be boolean")
    if not isinstance(unit, str):
        raise GroundingError("unit must be text")
    key = unit.strip().casefold()
    if key not in _UNIT_ALIASES:
        raise GroundingError(f"unsupported or unregistered measurement unit: {unit!r}")
    dimension, canonical_unit, factor, offset = _UNIT_ALIASES[key]
    converted = float(value) * factor + offset
    if not math.isfinite(converted):
        raise GroundingError("unit conversion produced a non-finite value")
    return converted, canonical_unit, dimension


def _parse_cell_value(value: Any, unit: str = "") -> tuple[Any, str, str]:
    if isinstance(value, bool):
        return value, "boolean", "boolean"
    if isinstance(value, (int, float)):
        number, canonical_unit, dimension = _normalise_numeric(value, unit)
        return number, canonical_unit, dimension
    if isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"([-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)\s*([^\s]*)", text)
        if match:
            try:
                number = float(match.group(1))
            except ValueError:
                number = float("nan")
            if math.isfinite(number):
                selected_unit = unit or match.group(2)
                number, canonical_unit, dimension = _normalise_numeric(number, selected_unit)
                return number, canonical_unit, dimension
        return text, "", "text"
    raise GroundingError("observation values must be string, boolean, or finite number")


@dataclass(frozen=True)
class AssetProvenance:
    source_id: str
    modality: Modality
    media_type: str
    sha256: str
    size_bytes: int
    source_uri: str
    retrieved_at: str
    publisher: str
    license_spdx: str
    integrity_verified: bool
    origin_verified: bool
    signer_id: str = ""
    signature_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modality"] = self.modality.value
        return value


@dataclass(frozen=True)
class GroundedObservation:
    observation_id: str
    source_id: str
    source_sha256: str
    modality: Modality
    entity_key: str
    attribute: str
    value: Any
    canonical_unit: str
    dimension: str
    locator: str
    observed_at: str
    integrity_verified: bool
    origin_verified: bool
    derivation: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["modality"] = self.modality.value
        return value


@dataclass(frozen=True)
class ExtractedSegment:
    locator: str
    text: str
    segment_kind: str = "text"

    def __post_init__(self) -> None:
        _required_text(self.locator, "segment locator", 512)
        _required_text(self.text, "segment text", 100_000)
        _required_text(self.segment_kind, "segment_kind", 64)


@dataclass(frozen=True)
class ModalConclusion:
    entity_key: str
    attribute: str
    status: GroundingStatus
    modalities: tuple[str, ...]
    source_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    values: tuple[Any, ...]
    canonical_unit: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["modalities"] = list(self.modalities)
        value["source_ids"] = list(self.source_ids)
        value["observation_ids"] = list(self.observation_ids)
        value["values"] = list(self.values)
        return value


@dataclass(frozen=True)
class GroundingReport:
    assets: tuple[AssetProvenance, ...]
    observations: tuple[GroundedObservation, ...]
    conclusions: tuple[ModalConclusion, ...]
    unknowns: tuple[str, ...]
    report_fingerprint: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets": [item.to_dict() for item in self.assets],
            "observations": [item.to_dict() for item in self.observations],
            "conclusions": [item.to_dict() for item in self.conclusions],
            "unknowns": list(self.unknowns),
            "report_fingerprint": self.report_fingerprint,
            "generated_at": self.generated_at,
        }


class SourceTrustRegistry:
    """Allow-list of Ed25519 source/device/perception identities.

    Registering a public key is a host trust decision. This registry verifies
    signatures; it does not decide whether a device or annotator deserves trust.
    """

    def __init__(self) -> None:
        self._keys: dict[str, bytes] = {}

    def register(self, identity_id: str, public_key: bytes) -> None:
        identity = _required_text(identity_id, "identity_id", 128)
        if not _IDENTITY.fullmatch(identity):
            raise GroundingError("identity_id contains unsupported characters")
        if not isinstance(public_key, bytes) or len(public_key) != 32:
            raise GroundingError("Ed25519 public keys must be exactly 32 bytes")
        if identity in self._keys:
            raise GroundingError(f"identity {identity!r} is already registered")
        self._keys[identity] = bytes(public_key)

    def verify(self, identity_id: str | None, signature_b64: str | None, payload: bytes) -> bool:
        if identity_id is None and signature_b64 is None:
            return False
        if not identity_id or not signature_b64:
            raise GroundingError("signer identity and signature must be supplied together")
        key = self._keys.get(identity_id)
        if key is None:
            raise GroundingError(f"signer identity {identity_id!r} is not trusted by this host")
        if not isinstance(signature_b64, str) or len(signature_b64) > 256:
            raise GroundingError("signature must be bounded base64 text")
        try:
            signature = base64.b64decode(signature_b64, validate=True)
            Ed25519PublicKey.from_public_bytes(key).verify(signature, payload)
        except (ValueError, binascii.Error, InvalidSignature) as exc:
            raise GroundingError("source signature verification failed") from exc
        return True


class RealWorldGroundingEngine:
    """Ingest bounded real-world evidence and compare cross-modal observations.

    Integrity means the received bytes match the parser's recorded digest and
    structural checks. Origin authenticity is true only after a trusted Ed25519
    signature verifies. The engine never upgrades unsigned content to trusted.
    """

    MAX_ASSETS = 128
    MAX_SOURCE_BYTES = 20 * 1024 * 1024
    MAX_TOTAL_SOURCE_BYTES = 64 * 1024 * 1024
    MAX_OBSERVATIONS = 10_000
    MAX_SEGMENTS = 10_000
    MAX_EXTRACTED_TEXT_BYTES = 4 * 1024 * 1024
    MAX_DOCX_ENTRIES = 128

    def __init__(self, trust_registry: SourceTrustRegistry | None = None) -> None:
        self.trust_registry = trust_registry or SourceTrustRegistry()
        self._assets: dict[str, AssetProvenance] = {}
        self._observations: dict[str, GroundedObservation] = {}
        self._source_bytes_total = 0

    @property
    def assets(self) -> tuple[AssetProvenance, ...]:
        return tuple(self._assets[key] for key in sorted(self._assets))

    @property
    def observations(self) -> tuple[GroundedObservation, ...]:
        return tuple(self._observations[key] for key in sorted(self._observations))

    def _register_asset(
        self,
        source_id: str,
        modality: Modality,
        media_type: str,
        raw: bytes,
        *,
        source_uri: str | None,
        retrieved_at: str | None,
        publisher: str,
        license_spdx: str,
        signer_id: str | None,
        signature_b64: str | None,
    ) -> AssetProvenance:
        source_id = _required_text(source_id, "source_id", 128)
        if not _IDENTITY.fullmatch(source_id):
            raise GroundingError("source_id contains unsupported characters")
        if source_id in self._assets:
            raise GroundingError(f"duplicate source_id: {source_id}")
        if len(self._assets) >= self.MAX_ASSETS:
            raise GroundingError("asset count limit exceeded")
        if not isinstance(raw, bytes) or not raw:
            raise GroundingError("source payload must be non-empty bytes")
        if len(raw) > self.MAX_SOURCE_BYTES:
            raise GroundingError("source exceeds per-asset byte limit")
        if self._source_bytes_total + len(raw) > self.MAX_TOTAL_SOURCE_BYTES:
            raise GroundingError("total source byte limit exceeded")
        digest = _digest(raw)
        origin_verified = self.trust_registry.verify(signer_id, signature_b64, raw)
        publisher = _required_text(publisher, "publisher", 256)
        license_spdx = _required_text(license_spdx, "license_spdx", 128)
        if not (license_spdx in {
            "0BSD", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0",
            "CC-BY-4.0", "ISC", "MIT", "MPL-2.0", "OFL-1.1", "PSF-2.0", "Unlicense", "Zlib",
        } or re.fullmatch(r"LicenseRef-[A-Za-z0-9.-]+", license_spdx)):
            raise GroundingError("license_spdx must be a recognized SPDX ID or LicenseRef")
        asset = AssetProvenance(
            source_id=source_id,
            modality=modality,
            media_type=_required_text(media_type, "media_type", 128),
            sha256=digest,
            size_bytes=len(raw),
            source_uri=_source_uri(source_uri, digest),
            retrieved_at=_utc_timestamp(retrieved_at, "retrieved_at"),
            publisher=publisher,
            license_spdx=license_spdx,
            integrity_verified=True,
            origin_verified=origin_verified,
            signer_id=signer_id or "",
            signature_sha256=_digest(base64.b64decode(signature_b64)) if signature_b64 else "",
        )
        self._assets[source_id] = asset
        self._source_bytes_total += len(raw)
        return asset

    def _add_observation(
        self,
        asset: AssetProvenance,
        *,
        entity_key: str,
        attribute: str,
        value: Any,
        unit: str = "",
        locator: str,
        observed_at: str | None = None,
        origin_verified: bool | None = None,
        derivation: str,
    ) -> GroundedObservation:
        if len(self._observations) >= self.MAX_OBSERVATIONS:
            raise GroundingError("observation count limit exceeded")
        entity = _required_text(entity_key, "entity_key", 256).casefold()
        attr = _required_text(attribute, "attribute", 256).casefold()
        location = _required_text(locator, "locator", 512)
        normalized, canonical_unit, dimension = _parse_cell_value(value, unit)
        stamp = _utc_timestamp(observed_at or asset.retrieved_at, "observed_at")
        payload = {
            "source_id": asset.source_id,
            "source_sha256": asset.sha256,
            "modality": asset.modality.value,
            "entity_key": entity,
            "attribute": attr,
            "value": normalized,
            "canonical_unit": canonical_unit,
            "dimension": dimension,
            "locator": location,
            "observed_at": stamp,
            "derivation": derivation,
        }
        observation_id = _digest(_canonical(payload).encode("utf-8"))
        if observation_id in self._observations:
            raise GroundingError("duplicate observation identity")
        observation = GroundedObservation(
            observation_id=observation_id,
            source_id=asset.source_id,
            source_sha256=asset.sha256,
            modality=asset.modality,
            entity_key=entity,
            attribute=attr,
            value=normalized,
            canonical_unit=canonical_unit,
            dimension=dimension,
            locator=location,
            observed_at=stamp,
            integrity_verified=asset.integrity_verified,
            origin_verified=asset.origin_verified if origin_verified is None else bool(origin_verified),
            derivation=_required_text(derivation, "derivation", 256),
        )
        self._observations[observation_id] = observation
        return observation

    @staticmethod
    def _verify_png_structure(raw: bytes) -> None:
        signature = b"\x89PNG\r\n\x1a\n"
        if len(raw) < 45 or raw[:8] != signature:
            raise PNGSecurityError("invalid PNG signature or truncated file")
        pos = 8
        chunks = 0
        saw_ihdr = False
        saw_idat = False
        idat_closed = False
        saw_iend = False
        width = height = 0
        bit_depth = color_type = 0
        compressed = bytearray()
        while pos < len(raw):
            chunks += 1
            if chunks > 4096 or pos + 12 > len(raw):
                raise PNGSecurityError("PNG chunk count/length limit exceeded")
            length, kind = struct.unpack(">I4s", raw[pos:pos + 8])
            pos += 8
            if length > len(raw) - pos - 4:
                raise PNGSecurityError("truncated PNG chunk payload")
            payload = raw[pos:pos + length]
            expected_crc = struct.unpack(">I", raw[pos + length:pos + length + 4])[0]
            actual_crc = zlib_crc32(kind + payload)
            if expected_crc != actual_crc:
                raise PNGSecurityError(f"PNG CRC mismatch in {kind.decode('ascii', 'replace')}")
            pos += length + 4
            if not saw_ihdr and kind != b"IHDR":
                raise PNGSecurityError("IHDR must be the first PNG chunk")
            if kind == b"IHDR":
                if saw_ihdr or length != 13:
                    raise PNGSecurityError("duplicate or malformed IHDR")
                width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
                if compression != 0 or filtering != 0 or interlace != 0:
                    raise PNGSecurityError("unsupported PNG compression/filter/interlace")
                if width <= 0 or height <= 0 or width > 2048 or height > 2048 or width * height > 4_194_304:
                    raise PNGSecurityError("PNG dimensions exceed the 2048x2048 safety cap")
                if bit_depth != 8 or color_type not in (0, 2, 4, 6):
                    raise PNGSecurityError("unsupported PNG color type or bit depth")
                saw_ihdr = True
            elif kind == b"IDAT":
                if idat_closed:
                    raise PNGSecurityError("PNG IDAT chunks must be consecutive")
                saw_idat = True
                if len(compressed) + length > SensoryCodec.MAX_IDAT_COMPRESSED_BYTES:
                    raise PNGSecurityError("PNG compressed payload limit exceeded")
                compressed.extend(payload)
            elif kind == b"IEND":
                if length != 0 or not saw_idat:
                    raise PNGSecurityError("malformed IEND or missing IDAT")
                saw_iend = True
                if pos != len(raw):
                    raise PNGSecurityError("trailing bytes after PNG IEND")
                break
            else:
                if saw_idat:
                    idat_closed = True
                if kind[0] & 0x20 == 0 and kind not in (b"PLTE",):
                    raise PNGSecurityError("unknown critical PNG chunk")
                if kind == b"PLTE":
                    raise PNGSecurityError("palette PNG is unsupported by the bounded grayscale/RGB decoder")
        if not (saw_ihdr and saw_idat and saw_iend):
            raise PNGSecurityError("PNG is missing a required IHDR, IDAT, or IEND chunk")
        channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
        expected_raw = height * (1 + width * channels)
        if expected_raw > SensoryCodec.MAX_DECOMPRESSED_BYTES:
            raise PNGSecurityError("PNG decompressed output exceeds safety limit")
        decompressor = zlib_decompressor()
        decoded = decompressor.decompress(bytes(compressed), expected_raw + 1)
        if decompressor.unconsumed_tail or len(decoded) != expected_raw or not decompressor.eof or decompressor.unused_data:
            raise PNGSecurityError("PNG zlib stream length, completion, or trailing-data check failed")

    def ingest_png(
        self,
        source_id: str,
        raw: bytes,
        *,
        entity_key: str | None = None,
        source_uri: str | None = None,
        retrieved_at: str | None = None,
        publisher: str = "host-supplied",
        license_spdx: str = "LicenseRef-UserSupplied",
        signer_id: str | None = None,
        signature_b64: str | None = None,
    ) -> AssetProvenance:
        if not isinstance(raw, bytes) or len(raw) > 10 * 1024 * 1024:
            raise GroundingError("PNG must be bytes and remain under the 10 MB file cap")
        self._verify_png_structure(raw)
        grid = SensoryCodec.decode_png_bytes_to_luma_grid(raw)
        width = len(grid[0])
        height = len(grid)
        flattened = [value for row in grid for value in row]
        mean_luma = sum(flattened) / len(flattened)
        variance = sum((value - mean_luma) ** 2 for value in flattened) / len(flattened)
        asset = self._register_asset(
            source_id, Modality.IMAGE, "image/png", raw, source_uri=source_uri,
            retrieved_at=retrieved_at, publisher=publisher, license_spdx=license_spdx,
            signer_id=signer_id, signature_b64=signature_b64,
        )
        entity = entity_key or f"image:{source_id}"
        features = (
            ("width_px", width, "px"),
            ("height_px", height, "px"),
            ("pixel_count", width * height, ""),
            ("mean_luma", mean_luma, "luma_0_255"),
            ("min_luma", min(flattened), "luma_0_255"),
            ("max_luma", max(flattened), "luma_0_255"),
            ("luma_variance", variance, "luma_0_255"),
        )
        for attribute, value, unit in features:
            self._add_observation(
                asset, entity_key=entity, attribute=attribute, value=value, unit=unit,
                locator="png:decoded-pixel-statistics", derivation="SensoryCodec PNG pixel statistics",
            )
        return asset

    def add_signed_image_annotation(
        self,
        source_id: str,
        *,
        entity_key: str,
        attribute: str,
        value: Any,
        unit: str = "",
        locator: str,
        annotator_id: str,
        signature_b64: str,
        observed_at: str,
    ) -> GroundedObservation:
        """Add a host perception result signed over the exact source image hash and annotation."""
        asset = self._assets.get(source_id)
        if asset is None or asset.modality is not Modality.IMAGE:
            raise GroundingError("image annotation must reference an ingested image asset")
        payload = {
            "source_id": asset.source_id,
            "source_sha256": asset.sha256,
            "entity_key": _required_text(entity_key, "entity_key", 256).casefold(),
            "attribute": _required_text(attribute, "attribute", 256).casefold(),
            "value": value,
            "unit": unit,
            "locator": _required_text(locator, "locator", 512),
            "observed_at": _utc_timestamp(observed_at, "observed_at"),
        }
        canonical = _canonical(payload).encode("utf-8")
        self.trust_registry.verify(annotator_id, signature_b64, canonical)
        return self._add_observation(
            asset, entity_key=payload["entity_key"], attribute=payload["attribute"],
            value=value, unit=unit, locator=payload["locator"], observed_at=payload["observed_at"],
            origin_verified=True, derivation=f"signed-perception-adapter:{annotator_id}",
        )

    @staticmethod
    def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GroundingError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise GroundingError(f"non-finite JSON number is forbidden: {value}")

    @staticmethod
    def _xml_text(element: ET.Element) -> str:
        return " ".join(part.strip() for part in element.itertext() if part.strip())

    def _docx_segments(self, raw: bytes) -> list[ExtractedSegment]:
        try:
            archive = zipfile.ZipFile(io.BytesIO(raw))
        except (zipfile.BadZipFile, OSError) as exc:
            raise GroundingError("DOCX is not a valid ZIP-based Office document") from exc
        with archive:
            items = archive.infolist()
            if len(items) > self.MAX_DOCX_ENTRIES:
                raise GroundingError("DOCX entry count limit exceeded")
            names = [item.filename for item in items]
            if len(set(names)) != len(names):
                raise GroundingError("DOCX contains duplicate ZIP entry names")
            total = 0
            for item in items:
                if item.is_dir():
                    continue
                if item.flag_bits & 0x1:
                    raise GroundingError("encrypted DOCX entries are not supported")
                if item.file_size > 4 * 1024 * 1024:
                    raise GroundingError("DOCX entry exceeds decompressed size cap")
                total += item.file_size
                if total > 8 * 1024 * 1024:
                    raise GroundingError("DOCX total decompressed size cap exceeded")
                if item.file_size > 0 and item.compress_size == 0:
                    raise GroundingError("DOCX suspicious zero-byte compressed entry")
                if item.compress_size and item.file_size / item.compress_size > 100:
                    raise GroundingError("DOCX compression ratio cap exceeded")
                path = item.filename.replace("\\", "/")
                if path.startswith("/") or any(part == ".." for part in path.split("/")):
                    raise GroundingError("DOCX archive contains unsafe path entries")
            try:
                xml = archive.read("word/document.xml")
            except KeyError as exc:
                raise GroundingError("DOCX is missing word/document.xml") from exc
        if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
            raise GroundingError("DOCX XML declarations/entities are forbidden")
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            raise GroundingError("DOCX document XML is malformed") from exc
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        body = root.find("w:body", ns)
        if body is None:
            raise GroundingError("DOCX document body is missing")
        segments: list[ExtractedSegment] = []
        paragraph_index = 0
        table_index = 0
        for child in list(body):
            if child.tag.endswith("}p"):
                paragraph_index += 1
                text = self._xml_text(child)
                if text:
                    segments.append(ExtractedSegment(f"paragraph:{paragraph_index}", text, "paragraph"))
            elif child.tag.endswith("}tbl"):
                table_index += 1
                for row_index, row in enumerate(child.findall("w:tr", ns), start=1):
                    cells = [self._xml_text(cell) for cell in row.findall("w:tc", ns)]
                    segments.append(ExtractedSegment(
                        f"table:{table_index}:row:{row_index}", " | ".join(cells), "table_row",
                    ))
        return segments

    @staticmethod
    def _csv_segments(text: str) -> tuple[list[ExtractedSegment], list[list[str]]]:
        try:
            rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
        except csv.Error as exc:
            raise GroundingError(f"CSV parse error: {exc}") from exc
        if not rows:
            raise GroundingError("CSV document contains no rows")
        if len(rows) > 10_000:
            raise GroundingError("CSV row count limit exceeded")
        width = max((len(row) for row in rows), default=0)
        if width > 256:
            raise GroundingError("CSV column count limit exceeded")
        segments = []
        for index, row in enumerate(rows, start=1):
            if len(row) > 256:
                raise GroundingError("CSV row exceeds column count limit")
            segments.append(ExtractedSegment(f"csv:row:{index}", " | ".join(row), "table_row"))
        return segments, rows

    def _structured_rows(self, source_id: str, asset: AssetProvenance, rows: Iterable[Mapping[str, Any]],
                         entity_default: str, locator_prefix: str) -> None:
        for index, row in enumerate(rows, start=1):
            lowered = {str(key).strip().casefold(): value for key, value in row.items()}
            if not {"attribute", "value"}.issubset(lowered):
                continue
            entity = lowered.get("entity_key", lowered.get("entity_id", entity_default))
            unit = lowered.get("unit", "")
            timestamp = lowered.get("observed_at", lowered.get("timestamp"))
            if not isinstance(entity, str) or not isinstance(lowered["attribute"], str):
                raise GroundingError("structured entity_key and attribute fields must be strings")
            if timestamp is not None and not isinstance(timestamp, str):
                raise GroundingError("structured observed_at/timestamp must be text")
            self._add_observation(
                asset, entity_key=entity, attribute=lowered["attribute"], value=lowered["value"],
                unit=str(unit), locator=f"{locator_prefix}:record:{index}",
                observed_at=timestamp, derivation="structured-document-record",
            )

    def ingest_document(
        self,
        source_id: str,
        filename: str,
        raw: bytes,
        *,
        entity_key: str | None = None,
        source_uri: str | None = None,
        retrieved_at: str | None = None,
        publisher: str = "host-supplied",
        license_spdx: str = "LicenseRef-UserSupplied",
        signer_id: str | None = None,
        signature_b64: str | None = None,
        extractor: Callable[[bytes], Iterable[ExtractedSegment]] | None = None,
        extractor_id: str | None = None,
    ) -> tuple[AssetProvenance, tuple[ExtractedSegment, ...]]:
        filename = _required_text(filename, "filename", 512)
        suffix = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
        if not isinstance(raw, bytes):
            raise GroundingError("document payload must be bytes")
        if len(raw) > self.MAX_SOURCE_BYTES:
            raise GroundingError("document exceeds per-asset byte limit")
        entity_default = entity_key or f"document:{source_id}"
        segments: list[ExtractedSegment]
        structured: list[Mapping[str, Any]] = []
        if suffix in {"txt", "md"}:
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise GroundingError("text document must be valid UTF-8") from exc
            segments = [
                ExtractedSegment(f"line:{index}", line, "line")
                for index, line in enumerate(text.splitlines(), start=1) if line.strip()
            ]
            if len(segments) > self.MAX_SEGMENTS:
                raise GroundingError("document segment count limit exceeded")
        elif suffix == "csv":
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise GroundingError("CSV document must be valid UTF-8") from exc
            segments, rows = self._csv_segments(text)
            if len(rows) > 1:
                headers = [str(item).strip().casefold() for item in rows[0]]
                structured = [
                    {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
                    for row in rows[1:]
                ]
        elif suffix == "json":
            try:
                text = raw.decode("utf-8", errors="strict")
                parsed = json.loads(
                    text, object_pairs_hook=self._json_no_duplicates,
                    parse_constant=self._reject_json_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GroundingError("JSON document is malformed or not UTF-8") from exc
            segments = [ExtractedSegment("json:root", _canonical(parsed), "structured_json")]
            if isinstance(parsed, dict):
                records = parsed.get("records", parsed.get("measurements", []))
                if isinstance(records, list) and all(isinstance(item, Mapping) for item in records):
                    structured = records
                elif {"attribute", "value"}.issubset(parsed):
                    structured = [parsed]
            elif isinstance(parsed, list) and all(isinstance(item, Mapping) for item in parsed):
                structured = parsed
        elif suffix == "docx":
            segments = self._docx_segments(raw)
        else:
            if extractor is None or not callable(extractor):
                raise GroundingError(
                    "unsupported document type; PDF and other formats require an explicit host extractor adapter"
                )
            extractor_id = _required_text(extractor_id, "extractor_id", 128)
            supplied = extractor(raw)
            segments = []
            try:
                for item in supplied:
                    if len(segments) >= self.MAX_SEGMENTS:
                        raise GroundingError("extractor segment count limit exceeded")
                    if not isinstance(item, ExtractedSegment):
                        raise GroundingError("extractor must yield ExtractedSegment records")
                    segments.append(item)
            except TypeError as exc:
                raise GroundingError("extractor must yield an iterable of ExtractedSegment records") from exc
        if not segments:
            raise GroundingError("document extraction produced no evidence segments")
        text_size = sum(len(item.text.encode("utf-8")) for item in segments)
        if text_size > self.MAX_EXTRACTED_TEXT_BYTES:
            raise GroundingError("extracted document text exceeds the 4 MB limit")
        media_type = {
            "txt": "text/plain", "md": "text/markdown", "csv": "text/csv",
            "json": "application/json", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }.get(suffix, "application/octet-stream")
        asset = self._register_asset(
            source_id, Modality.DOCUMENT, media_type, raw, source_uri=source_uri,
            retrieved_at=retrieved_at, publisher=publisher, license_spdx=license_spdx,
            signer_id=signer_id, signature_b64=signature_b64,
        )
        if structured:
            self._structured_rows(source_id, asset, structured, entity_default, "document")
        return asset, tuple(segments)

    @staticmethod
    def telemetry_signing_bytes(sample: Mapping[str, Any]) -> bytes:
        """Canonical bytes a device signs; signature field itself is never included."""
        required = ("sample_id", "entity_key", "attribute", "value", "unit", "observed_at")
        if not isinstance(sample, Mapping) or any(key not in sample for key in required):
            raise GroundingError("telemetry signing payload requires sample_id/entity_key/attribute/value/unit/observed_at")
        payload = {key: sample[key] for key in required}
        payload["sample_id"] = _required_text(payload["sample_id"], "sample_id", 128)
        payload["entity_key"] = _required_text(payload["entity_key"], "entity_key", 256).casefold()
        payload["attribute"] = _required_text(payload["attribute"], "attribute", 256).casefold()
        payload["observed_at"] = _utc_timestamp(payload["observed_at"], "observed_at")
        _parse_cell_value(payload["value"], str(payload["unit"]))
        return _canonical(payload).encode("utf-8")

    def ingest_telemetry(
        self,
        source_id: str,
        sample: Mapping[str, Any],
        *,
        device_id: str,
        signature_b64: str | None = None,
        source_uri: str | None = None,
        publisher: str = "host-supplied-device-gateway",
    ) -> tuple[AssetProvenance, GroundedObservation]:
        if not isinstance(sample, Mapping):
            raise GroundingError("telemetry sample must be a mapping")
        required = {"sample_id", "entity_key", "attribute", "value", "unit", "observed_at"}
        if not required.issubset(sample):
            raise GroundingError(f"telemetry sample missing fields: {sorted(required - set(sample))}")
        clean = {
            "sample_id": sample["sample_id"],
            "entity_key": sample["entity_key"],
            "attribute": sample["attribute"],
            "value": sample["value"],
            "unit": sample["unit"],
            "observed_at": sample["observed_at"],
        }
        signing_payload = self.telemetry_signing_bytes(clean)
        device_id = _required_text(device_id, "device_id", 128)
        verified = self.trust_registry.verify(device_id, signature_b64, signing_payload)
        raw = signing_payload
        asset = self._register_asset(
            source_id, Modality.TELEMETRY, "application/vnd.mirror.telemetry+json", raw,
            source_uri=source_uri, retrieved_at=clean["observed_at"], publisher=publisher,
            license_spdx="LicenseRef-TelemetryRecord", signer_id=None, signature_b64=None,
        )
        # The signature authenticates the canonical record, not the raw asset digest.
        if verified:
            asset = AssetProvenance(
                **{**asdict(asset), "modality": Modality.TELEMETRY, "origin_verified": True,
                   "signer_id": device_id, "signature_sha256": _digest(base64.b64decode(signature_b64))},
            )
            self._assets[source_id] = asset
        observation = self._add_observation(
            asset, entity_key=str(clean["entity_key"]), attribute=str(clean["attribute"]),
            value=clean["value"], unit=str(clean["unit"]), locator=f"device:{device_id}:sample:{clean['sample_id']}",
            observed_at=str(clean["observed_at"]), derivation="signed-telemetry-record" if verified else "unsigned-telemetry-record",
        )
        return asset, observation

    @staticmethod
    def _compatible(a: GroundedObservation, b: GroundedObservation,
                    absolute_tolerance: float, relative_tolerance: float) -> tuple[bool | None, str]:
        if a.dimension != b.dimension:
            return None, "measurement dimensions differ"
        if a.dimension in {"text", "boolean"}:
            return a.value == b.value, "categorical values agree" if a.value == b.value else "categorical values differ"
        left = float(a.value)
        right = float(b.value)
        tolerance = max(absolute_tolerance, relative_tolerance * max(abs(left), abs(right)))
        agrees = abs(left - right) <= tolerance
        return agrees, (
            f"numeric values agree within tolerance {tolerance:g}"
            if agrees else f"numeric values differ by {abs(left-right):g}, tolerance {tolerance:g}"
        )

    def analyze(
        self,
        *,
        absolute_tolerance: float = 0.01,
        relative_tolerance: float = 0.005,
    ) -> GroundingReport:
        for value, name in ((absolute_tolerance, "absolute_tolerance"), (relative_tolerance, "relative_tolerance")):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
                raise GroundingError(f"{name} must be a finite non-negative number")
        groups: dict[tuple[str, str], list[GroundedObservation]] = {}
        for item in self.observations:
            groups.setdefault((item.entity_key, item.attribute), []).append(item)
        conclusions: list[ModalConclusion] = []
        unknowns: list[str] = []
        for (entity, attribute), items in sorted(groups.items()):
            modalities = tuple(sorted({item.modality.value for item in items}))
            sources = tuple(sorted({item.source_id for item in items}))
            observations = tuple(sorted({item.observation_id for item in items}))
            values = tuple(item.value for item in items)
            units = {item.canonical_unit for item in items}
            if len(modalities) < 2 or len(sources) < 2:
                status = GroundingStatus.INSUFFICIENT
                explanation = "only one independent source modality is available; cross-modal corroboration is not established"
                unknowns.append(f"{entity}.{attribute}: needs a second distinct source modality")
            elif len(units) != 1 or len({item.dimension for item in items}) != 1:
                status = GroundingStatus.INCOMPARABLE
                explanation = "observations use incompatible measurement units or value dimensions"
                unknowns.append(f"{entity}.{attribute}: observations cannot be compared safely")
            else:
                pair_results = [
                    self._compatible(items[i], items[j], float(absolute_tolerance), float(relative_tolerance))
                    for i in range(len(items)) for j in range(i + 1, len(items))
                ]
                if any(result is None for result, _ in pair_results):
                    status = GroundingStatus.INCOMPARABLE
                    explanation = "at least one observation pair is not comparable"
                    unknowns.append(f"{entity}.{attribute}: comparison is incomplete")
                elif not all(bool(result) for result, _ in pair_results):
                    status = GroundingStatus.CONFLICT
                    explanation = "cross-modal observations disagree: " + "; ".join(detail for _, detail in pair_results)
                elif all(item.integrity_verified and item.origin_verified for item in items):
                    status = GroundingStatus.VERIFIED_AGREEMENT
                    explanation = "distinct source modalities agree and every source origin has a valid trusted signature"
                else:
                    status = GroundingStatus.CORROBORATED_UNVERIFIED
                    explanation = "distinct source modalities agree, but one or more source origins are not authenticated"
                    unknowns.append(f"{entity}.{attribute}: agreement exists but source authenticity remains unverified")
            conclusions.append(ModalConclusion(
                entity_key=entity, attribute=attribute, status=status,
                modalities=modalities, source_ids=sources, observation_ids=observations,
                values=values, canonical_unit=next(iter(units)) if len(units) == 1 else "",
                explanation=explanation,
            ))
        report_payload = {
            "assets": [item.to_dict() for item in self.assets],
            "observations": [item.to_dict() for item in self.observations],
            "conclusions": [item.to_dict() for item in conclusions],
            "unknowns": unknowns,
        }
        fingerprint = _digest(_canonical(report_payload).encode("utf-8"))
        return GroundingReport(
            assets=self.assets, observations=self.observations, conclusions=tuple(conclusions),
            unknowns=tuple(unknowns), report_fingerprint=fingerprint,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


def zlib_crc32(data: bytes) -> int:
    import zlib
    return zlib.crc32(data) & 0xFFFFFFFF


def zlib_decompressor() -> Any:
    import zlib
    return zlib.decompressobj()
