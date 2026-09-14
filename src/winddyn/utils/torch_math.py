"""Batched quaternion / rotation utilities (torch, shape (N, ...) everywhere).

Quaternion convention: (w, x, y, z), unit norm, world-from-body ("R(q) maps
body vectors into the world frame") -- the same convention Isaac Lab uses.
"""

from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors v into the world frame."""
    qw = q[..., :1]
    qv = q[..., 1:]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + qw * t + torch.cross(qv, t, dim=-1)


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors v into the body frame."""
    return quat_rotate(quat_conj(q), v)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) -> (N, 3, 3) rotation matrices (world-from-body)."""
    w, x, y, z = quat_normalize(q).unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle.unsqueeze(-1)
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_integrate(q: torch.Tensor, omega_body: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order quaternion integration with body-frame angular velocity."""
    dq = torch.cat([torch.zeros_like(q[..., :1]), omega_body], dim=-1)
    q_new = q + 0.5 * dt * quat_mul(q, dq)
    return quat_normalize(q_new)


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) -> (N, 4) (w, x, y, z). Shepperd's method, batched.

    Branch selection on the largest component. The `copysign` shortcut is wrong
    for 180-degree rotations, where w == 0 and the sign is decided by the sign
    of a floating-point zero.
    """
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    eps = 1e-12

    def branch(s, w, x, y, z):
        return torch.stack([w, x, y, z], dim=-1)

    s0 = torch.sqrt(torch.clamp(tr + 1.0, min=eps)) * 2.0
    q0 = branch(s0, 0.25 * s0, (m21 - m12) / s0, (m02 - m20) / s0, (m10 - m01) / s0)

    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=eps)) * 2.0
    q1 = branch(s1, (m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1)

    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=eps)) * 2.0
    q2 = branch(s2, (m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2)

    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=eps)) * 2.0
    q3 = branch(s3, (m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3)

    use0 = (tr > 0.0).unsqueeze(-1)
    use1 = ((m00 > m11) & (m00 > m22)).unsqueeze(-1)
    use2 = (m11 > m22).unsqueeze(-1)
    q = torch.where(use0, q0, torch.where(use1, q1, torch.where(use2, q2, q3)))
    q = quat_normalize(q)
    # canonical hemisphere (q and -q are the same rotation)
    return torch.where(q[..., :1] < 0.0, -q, q)


def tilt_angle(q: torch.Tensor) -> torch.Tensor:
    """Angle (rad) between the body +Z axis and world +Z. (N,)"""
    z_body = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    z_body[..., 2] = 1.0
    z_world = quat_rotate(q, z_body)
    return torch.acos(z_world[..., 2].clamp(-1.0, 1.0))


def yaw_of_axis(axis_world: torch.Tensor) -> torch.Tensor:
    """Heading angle (rad) of a world-frame direction, atan2(y, x). (N,)"""
    return torch.atan2(axis_world[..., 1], axis_world[..., 0])


def vee(m: torch.Tensor) -> torch.Tensor:
    """Inverse hat map: (N, 3, 3) skew -> (N, 3)."""
    return torch.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], dim=-1)
