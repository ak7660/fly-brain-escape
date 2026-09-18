"""Looming threats → drive for LPLC2 (angular size) and LC4 (angular velocity) cells.

Side-only encoding: each eye has one broad Gaussian receptive field (config.RF_*). This file is
mirrored function by function in web/stimulus.js; keep the two in sync.
"""
from dataclasses import dataclass

import numpy as np

from flybrain import config


@dataclass(frozen=True)
class Threat:
    azimuth_deg: float
    lv: float  # half-size / approach speed (s)
    t_spawn: float  # s
    t_collision: float  # s


def loom_angle_deg(t, lv, t_collision):
    return np.degrees(2.0 * np.arctan(lv / np.maximum(np.asarray(t_collision) - t, 1e-9)))


def loom_rate_dps(t, lv, t_collision):
    dt = np.maximum(np.asarray(t_collision) - t, 1e-9)
    return np.degrees(2.0 * lv / (dt * dt + lv * lv))


def rf_gain(azimuth_deg, side):
    center = np.asarray(config.RF_CENTERS_DEG)[np.asarray(side) - 1]
    d = (np.asarray(azimuth_deg, dtype=float) - center + 180.0) % 360.0 - 180.0
    return np.exp(-0.5 * (d / config.RF_SIGMA_DEG) ** 2)


def _kind_drive(t, th: Threat):
    if not (th.t_spawn <= t < th.t_collision):
        return 0.0, 0.0
    size = min(loom_angle_deg(t, th.lv, th.t_collision) / config.THETA_SAT_DEG, 1.0)
    speed = min(loom_rate_dps(t, th.lv, th.t_collision) / config.THETADOT_SAT_DPS, 1.0)
    return float(size), float(speed)


def input_drive(t, threats, input_kind, input_side):
    drive = np.zeros(len(input_kind))
    for th in threats:
        per_kind = _kind_drive(t, th)
        if per_kind == (0.0, 0.0):
            continue
        drive += np.asarray(per_kind)[input_kind] * rf_gain(th.azimuth_deg, input_side)
    return drive


def threat_class(azimuth_deg):
    az = (azimuth_deg + 180.0) % 360.0 - 180.0
    if abs(az) <= config.FRONT_HALF_WIDTH_DEG:
        return 3
    return 1 if az < 0 else 2


def frame_labels(threats, n_frames):
    labels = np.zeros(n_frames, dtype=np.int64)
    best = np.zeros(n_frames)
    t = np.arange(n_frames) / config.FPS
    for th in threats:
        active = (t >= th.t_spawn) & (t < th.t_collision)
        theta = np.where(active, loom_angle_deg(t, th.lv, th.t_collision), 0.0)
        take = (theta >= config.LABEL_ONSET_DEG) & (theta > best)
        labels[take] = threat_class(th.azimuth_deg)
        best = np.maximum(best, np.where(theta >= config.LABEL_ONSET_DEG, theta, 0.0))
    return labels


_CLASS_AZIMUTH = {1: -90.0, 2: 90.0, 3: 0.0}


def sample_episodes(n, input_kind, input_side, rng, n_frames=72, lv_ranges=((0.02, 0.06),), p_empty=0.2,
                    az_jitter_deg=20.0, gain_jitter=0.1, noise=config.NOISE_SD, onset_max_frame=24,
                    approach_s=config.APPROACH_S):
    input_kind, input_side = np.asarray(input_kind), np.asarray(input_side)
    ni = len(input_kind)
    t = np.arange(n_frames) / config.FPS
    drive = np.zeros((n, n_frames, ni), dtype=np.float32)
    labels = np.zeros((n, n_frames), dtype=np.int64)
    mask = np.ones((n, n_frames), dtype=bool)
    classes = np.where(rng.random(n) < p_empty, 0, rng.integers(1, 4, n))
    threats = []
    for i, cls in enumerate(classes):
        gains = 1.0 + gain_jitter * rng.standard_normal(ni)
        eps = []
        if cls:
            lo, hi = lv_ranges[rng.integers(len(lv_ranges))]
            # spawn and collision snap to whole frames so float times never straddle a frame boundary
            spawn_f = int(rng.integers(0, onset_max_frame + 1))
            collide_f = spawn_f + int(round(approach_s * config.FPS))
            az = _CLASS_AZIMUTH[int(cls)] + rng.uniform(-az_jitter_deg, az_jitter_deg)
            th = Threat(float(az), float(rng.uniform(lo, hi)), spawn_f / config.FPS, collide_f / config.FPS)
            eps = [th]
            for f in range(n_frames):
                drive[i, f] = input_drive(t[f], eps, input_kind, input_side) * gains
            labels[i] = frame_labels(eps, n_frames)
            mask[i] = np.arange(n_frames) < collide_f
        drive[i] += (noise * rng.standard_normal((n_frames, ni))).astype(np.float32)
        threats.append(eps)
    return {"drive": drive, "labels": labels, "mask": mask, "episode_class": classes.astype(np.int64),
            "threats": threats}
