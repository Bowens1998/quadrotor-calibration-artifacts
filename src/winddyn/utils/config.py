"""YAML config loading + the derived vehicle parameter set.

Adapted from the user's WindJEPA repository (MIT). The Starling 2 Max numbers
in configs/robot/starling_2_max.yaml were extracted/derived from the private
vendor USD by that project's audit tooling; the USD itself is never committed.
The private USD path, when needed by an Isaac backend, comes from the
STARLING_USD_PATH environment variable — the torch reference simulator does
not need it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

GRAVITY = 9.81  # m/s^2


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = repo_root() / p
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root() / p


@dataclass
class VehicleParams:
    """Numeric vehicle parameters, derived once from the robot YAML."""

    name: str
    usd_path: Path | None
    mass: float
    inertia_diag: np.ndarray          # (3,)
    com: np.ndarray                   # (3,)
    collision_extents: np.ndarray     # (3,) full extents, m
    rotor_positions: np.ndarray       # (4, 3) body frame, m
    spin_dirs: np.ndarray             # (4,) +1 CCW / -1 CW about +Z
    arm_length: float
    nose_axis_body: np.ndarray        # (3,)
    k_f: float
    k_m: float
    w_max: float
    max_rotor_thrust: float           # N, per rotor
    motor_time_constant: float        # s
    torque_to_thrust: float           # m
    air_density: float
    body_drag_cda: np.ndarray         # (3,)
    rotor_drag_coeff: np.ndarray      # (3,)
    rotor_drag_scale_with_thrust: bool
    angular_drag_coeff: np.ndarray    # (3,)
    tof: dict[str, Any] = field(default_factory=dict)
    domain_randomization: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def weight(self) -> float:
        return self.mass * GRAVITY

    @property
    def hover_thrust_per_rotor(self) -> float:
        return self.weight / len(self.rotor_positions)


def load_vehicle(path: str | Path = "configs/robot/starling_2_max.yaml") -> VehicleParams:
    cfg = load_yaml(path)

    rb = cfg["rigid_body"]
    inertia_block = rb["inertia"]
    source = inertia_block["source"]
    inertia = np.asarray(inertia_block[source], dtype=np.float64)
    if not np.all(inertia > 0):
        raise ValueError(f"inertia '{source}' is non-positive: {inertia}")

    rot = cfg["rotors"]
    prop = rot["propulsion"]
    mass = float(rb["mass"])
    n = len(rot["positions"])
    w_max = float(prop["max_speed"])
    k_f = float(prop["max_thrust_to_weight"]) * mass * GRAVITY / n / w_max**2
    k_m = k_f * float(prop["torque_to_thrust_ratio"])

    aero = cfg["aero"]
    if aero.get("status") != "UNIDENTIFIED_AERODYNAMICS":
        raise ValueError(
            "aero.status must be 'UNIDENTIFIED_AERODYNAMICS' until a real "
            "identification campaign has been done. Do not remove the tag."
        )

    usd = os.environ.get("STARLING_USD_PATH")
    tof_list = cfg.get("sensors", {}).get("depth", [])
    tof = tof_list[0] if tof_list else {}

    return VehicleParams(
        name=cfg["name"],
        usd_path=Path(usd) if usd else None,
        mass=mass,
        inertia_diag=inertia,
        com=np.asarray(rb["center_of_mass"], dtype=np.float64),
        collision_extents=np.asarray(rb["collision"]["full_extents"], dtype=np.float64),
        rotor_positions=np.asarray(rot["positions"], dtype=np.float64),
        spin_dirs=np.asarray(rot["spin_dirs"], dtype=np.float64),
        arm_length=float(rot["arm_length"]),
        nose_axis_body=np.asarray(cfg["frames"]["nose_axis_body"], dtype=np.float64),
        k_f=k_f,
        k_m=k_m,
        w_max=w_max,
        max_rotor_thrust=k_f * w_max**2,
        motor_time_constant=float(prop["time_constant"]),
        torque_to_thrust=float(prop["torque_to_thrust_ratio"]),
        air_density=float(aero["air_density"]),
        body_drag_cda=np.asarray(aero["body_drag"]["cda"], dtype=np.float64),
        rotor_drag_coeff=np.asarray(aero["rotor_drag"]["coeff"], dtype=np.float64),
        rotor_drag_scale_with_thrust=bool(aero["rotor_drag"]["scale_with_thrust"]),
        angular_drag_coeff=np.asarray(aero["angular_drag"]["coeff"], dtype=np.float64),
        tof=tof,
        domain_randomization=cfg.get("domain_randomization", {}),
        raw=cfg,
    )


def print_aero_warning() -> None:
    print(
        "\n"
        "=================================================================\n"
        "WARNING: UNIDENTIFIED_AERODYNAMICS\n"
        "Using randomized low-order drag model. Results are not a\n"
        "calibrated digital twin of the Starling 2 Max.\n"
        "=================================================================\n",
        flush=True,
    )
