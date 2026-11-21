"""Deterministic SI Unit Normalization Layer for Mirror 7 (Zero 3rd-Party Dependencies).

Converts common metric and imperial unit tokens attached to numeric parameters
into Mirror's canonical SI base units so UOL rules and the Unit-Graph Planner
handle real-world mixed-unit word problems with 100% exactness.
"""
from __future__ import annotations

import re

# Mapping: canonical quantity -> {unit_token_lowercase: multiplier_to_SI}
_UNIT_FACTORS: dict[str, dict[str, float]] = {
    "mass": {
        "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
        "g": 1e-3, "gram": 1e-3, "grams": 1e-3,
        "mg": 1e-6, "milligram": 1e-6, "milligrams": 1e-6,
        "t": 1000.0, "ton": 1000.0, "tonne": 1000.0, "tonnes": 1000.0,
        "lb": 0.45359237, "lbs": 0.45359237, "pound": 0.45359237, "pounds": 0.45359237,
        "oz": 0.028349523125, "ounce": 0.028349523125, "ounces": 0.028349523125,
    },
    "velocity": {
        "m/s": 1.0, "mps": 1.0,
        "km/h": 1.0 / 3.6, "kmh": 1.0 / 3.6, "kph": 1.0 / 3.6, "kmph": 1.0 / 3.6,
        "mph": 0.44704, "mi/h": 0.44704,
        "cm/s": 0.01,
        "ft/s": 0.3048,
    },
    "length": {
        "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
        "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
        "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
        "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
        "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
        "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
        "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    },
    "area": {
        "m^2": 1.0, "m2": 1.0, "m²": 1.0, "sqm": 1.0,
        "cm^2": 1e-4, "cm2": 1e-4, "cm²": 1e-4, "sqcm": 1e-4,
        "mm^2": 1e-6, "mm2": 1e-6, "mm²": 1e-6,
        "km^2": 1e6, "km2": 1e6, "km²": 1e6,
        "ft^2": 0.09290304, "ft2": 0.09290304, "sqft": 0.09290304,
    },
    "volume": {
        "m^3": 1.0, "m3": 1.0, "m³": 1.0,
        "l": 1e-3, "liter": 1e-3, "liters": 1e-3, "litre": 1e-3, "litres": 1e-3,
        "ml": 1e-6, "cm^3": 1e-6, "cm3": 1e-6, "cm³": 1e-6,
    },
    "liters": {
        "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
        "ml": 1e-3, "milliliter": 1e-3, "milliliters": 1e-3,
        "m^3": 1000.0, "m3": 1000.0,
    },
    "force": {
        "n": 1.0, "newton": 1.0, "newtons": 1.0,
        "kn": 1000.0, "kilonewton": 1000.0, "kilonewtons": 1000.0,
        "mn": 1e6, "meganewton": 1e6,
        "lbf": 4.4482216,
    },
    "pressure": {
        "pa": 1.0, "pascal": 1.0, "pascals": 1.0,
        "kpa": 1000.0, "kilopascal": 1000.0,
        "mpa": 1e6, "megapascal": 1e6,
        "gpa": 1e9, "gigapascal": 1e9,
        "bar": 1e5,
        "psi": 6894.757293,
    },
    "energy": {
        "j": 1.0, "joule": 1.0, "joules": 1.0,
        "kj": 1000.0, "kilojoule": 1000.0, "kilojoules": 1000.0,
        "mj": 1e6, "megajoule": 1e6,
        "wh": 3600.0, "kwh": 3.6e6,
        "cal": 4.184, "kcal": 4184.0,
    },
    "power": {
        "w": 1.0, "watt": 1.0, "watts": 1.0,
        "kw": 1000.0, "kilowatt": 1000.0, "kilowatts": 1000.0,
        "mw": 1e6, "megawatt": 1e6,
        "hp": 745.699872,
    },
    "current": {
        "a": 1.0, "amp": 1.0, "amps": 1.0, "ampere": 1.0, "amperes": 1.0,
        "ma": 1e-3, "milliamp": 1e-3, "milliamps": 1e-3,
        "ka": 1000.0,
    },
    "voltage": {
        "v": 1.0, "volt": 1.0, "volts": 1.0,
        "mv": 1e-3, "millivolt": 1e-3,
        "kv": 1000.0, "kilovolt": 1000.0,
    },
    "resistance": {
        "ohm": 1.0, "ohms": 1.0, "ω": 1.0,
        "kohm": 1000.0, "kohms": 1000.0, "kω": 1000.0,
        "mohm": 1e6, "mohms": 1e6, "mω": 1e6,
    },
    "time": {
        "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
        "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3,
        "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
        "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    },
    "frequency": {
        "hz": 1.0, "hertz": 1.0,
        "khz": 1000.0, "mhz": 1e6, "ghz": 1e9,
    },
}

# Map canonical quantity names to their physical dimension group
QUANTITY_TO_DIMENSION: dict[str, str] = {
    "mass": "mass",
    "mass_kg": "mass",
    "velocity": "velocity",
    "wave_speed": "velocity",
    "distance": "length",
    "height": "length",
    "height_m": "length",
    "radius": "length",
    "r": "length",
    "span": "length",
    "base": "length",
    "wavelength": "length",
    "area": "area",
    "circle_area": "area",
    "triangle_area": "area",
    "volume": "volume",
    "cylinder_volume": "volume",
    "liters": "liters",
    "force": "force",
    "load": "force",
    "stress": "pressure",
    "applied_stress": "pressure",
    "yield_strength": "pressure",
    "pressure": "pressure",
    "kinetic_energy": "energy",
    "potential_energy": "energy",
    "energy": "energy",
    "work": "energy",
    "input_energy": "energy",
    "output_energy": "energy",
    "electrical_power": "power",
    "current": "current",
    "voltage": "voltage",
    "resistance": "resistance",
    "time": "time",
    "frequency": "frequency",
}


def normalize_to_si(value: float, unit_token: str | None, quantity: str | None) -> tuple[float, bool]:
    """Convert `value` with optional `unit_token` into canonical SI for `quantity`.

    Returns `(converted_value, was_converted)`. If `unit_token` is empty or not
    a recognized unit for that quantity's dimension, returns `(value, False)`.
    """
    if not unit_token or not quantity:
        return value, False
    dim = QUANTITY_TO_DIMENSION.get(quantity.strip().casefold())
    if not dim:
        return value, False
    table = _UNIT_FACTORS.get(dim)
    if not table:
        return value, False
    clean_unit = re.sub(r"\s+", "", unit_token.strip().casefold())
    factor = table.get(clean_unit)
    if factor is None:
        return value, False
    return float(value) * factor, (factor != 1.0)


def unit_pattern_for_quantity(quantity: str | None) -> str:
    """Return a regex capturing group matching any known unit token for `quantity`."""
    if not quantity:
        return ""
    dim = QUANTITY_TO_DIMENSION.get(quantity.strip().casefold())
    if not dim:
        return ""
    tokens = sorted(_UNIT_FACTORS.get(dim, {}).keys(), key=len, reverse=True)
    if not tokens:
        return ""
    return "|".join(re.escape(t) for t in tokens)
