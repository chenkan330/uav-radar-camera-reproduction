"""Reproducible synthetic horizontal manoeuvres for V4 uncertainty experiments.

These are declared experiment assumptions, not the paper's mobility model.
Position and velocity are continuous: control changes never teleport a target.
The 8 m/s² bound is *tangential* acceleration; turns add centripetal acceleration.
Future event times and left/right choices are simulator labels, never inputs.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


MOTION_NAMES = {0: 'constant_speed', 1: 'acceleration_braking', 2: 'sharp_turn',
                3: 'sudden_maneuver', 4: 'mixed'}
MOTION_NAMES_ZH = {0: '匀速', 1: '加速与制动', 2: '急转弯', 3: '突然机动', 4: '混合运动'}
BASE_STATION = np.array([0., 0., 10.])


@dataclass(frozen=True)
class ManeuverConfig:
    minimum_speed_m_s: float = 10.
    maximum_speed_m_s: float = 35.
    maximum_tangential_acceleration_m_s2: float = 8.
    maximum_lateral_acceleration_m_s2: float = 12.
    maximum_turn_rate_deg_s: float = 120.
    minimum_event_duration_s: float = .75
    maximum_event_duration_s: float = 2.5
    steering_radius_m: float = 900.

    def validate(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Motion parameters must be finite and positive')
        if self.minimum_speed_m_s > self.maximum_speed_m_s:
            raise ValueError('Invalid speed interval')
        if self.minimum_event_duration_s > self.maximum_event_duration_s:
            raise ValueError('Invalid event-duration interval')
        return self


def centered_dft_beams(position):
    """The existing 32 x 32 half-bin centred DFT, 1-based on disk."""
    rel = np.asarray(position) - BASE_STATION
    distance = np.maximum(np.linalg.norm(rel, axis=-1), 1e-8)
    ij = np.clip(np.floor((rel[..., :2]/distance[..., None]+1)*16).astype(np.int64), 0, 31)
    return ij[..., 1]*32+ij[..., 0]+1


def _kind_id(kind):
    if isinstance(kind, str):
        inverse = {name: value for value, name in MOTION_NAMES.items()}
        if kind not in inverse:
            raise ValueError(f'Unknown movement type: {kind}')
        return inverse[kind]
    if isinstance(kind, (int, np.integer)) and int(kind) in MOTION_NAMES:
        return int(kind)
    raise ValueError(f'Unknown movement type: {kind}')


def _controls(rng, cohort, prior_regime, event_index, config):
    regime = cohort
    if cohort == 4:
        options = [r for r in range(4) if r != prior_regime]
        regime = int(rng.choice(options))
    if regime == 0:
        acceleration, turn_rate = 0., 0.
    elif regime == 1:
        sign = (1. if event_index % 2 == 0 else -1.)
        acceleration = sign*rng.uniform(.25*config.maximum_tangential_acceleration_m_s2,
                                       config.maximum_tangential_acceleration_m_s2)
        turn_rate = 0.
    elif regime == 2:
        acceleration = 0.
        turn_rate = math.radians(rng.uniform(.375*config.maximum_turn_rate_deg_s,
                                            config.maximum_turn_rate_deg_s))*rng.choice((-1., 1.))
    else:
        # A sudden control onset can be either longitudinal or lateral; during
        # other intervals the target continues straight. No direction is given
        # to a predictor before the onset.
        action = int(rng.integers(0, 3))
        acceleration = rng.uniform(-config.maximum_tangential_acceleration_m_s2,
                                   config.maximum_tangential_acceleration_m_s2) if action != 0 else 0.
        turn_rate = (math.radians(rng.uniform(.5*config.maximum_turn_rate_deg_s, config.maximum_turn_rate_deg_s))*
                     rng.choice((-1., 1.))) if action == 2 else 0.
    return regime, float(acceleration), float(turn_rate)


def _reference_pool(initial_references):
    if initial_references is None:
        return None, None
    if isinstance(initial_references, dict):
        positions = np.asarray(initial_references['position'], dtype=float)
        velocities = initial_references.get('velocity')
        velocities = np.asarray(velocities, dtype=float) if velocities is not None else None
    else:
        positions, velocities = np.asarray(initial_references, dtype=float), None
    if positions.ndim not in (2, 3) or positions.shape[-1] != 3 or not positions.size:
        raise ValueError('Reference positions require [pool,3] or [trajectory,time,3]')
    if not np.isfinite(positions).all():
        raise ValueError('Reference positions must be finite')
    if velocities is not None and (velocities.shape != positions.shape or not np.isfinite(velocities).all()):
        raise ValueError('Reference velocity shape/values differ from positions')
    return positions.reshape(-1, 3), None if velocities is None else velocities.reshape(-1, 3)


def _initial_state(rng, positions, velocities, config):
    if positions is None:
        radius, angle = rng.uniform(150., 400.), rng.uniform(-np.pi, np.pi)
        position = np.array([radius*np.cos(angle), radius*np.sin(angle), rng.uniform(50., 100.)])
        direction = rng.uniform(-np.pi, np.pi)
    else:
        index = int(rng.integers(len(positions)))
        position = positions[index].copy()
        source_velocity = None if velocities is None else velocities[index]
        direction = (math.atan2(source_velocity[1], source_velocity[0])
                     if source_velocity is not None and np.linalg.norm(source_velocity[:2]) > 1e-6
                     else rng.uniform(-np.pi, np.pi))
    speed = rng.uniform(config.minimum_speed_m_s, config.maximum_speed_m_s)
    velocity = speed*np.array([np.cos(direction), np.sin(direction), 0.])
    return position, velocity


def _advance(position, velocity, acceleration, turn_rate, dt, config):
    speed = float(np.linalg.norm(velocity[:2]))
    heading = math.atan2(velocity[1], velocity[0])
    if math.hypot(position[0], position[1]) >= config.steering_radius_m:
        # Smooth steering, no coordinate clipping or position reset. It depends
        # only on the simulator's current state, not a future path.
        desired = math.atan2(-position[1], -position[0])
        difference = (desired-heading+np.pi) % (2*np.pi)-np.pi
        turn_rate = min(math.radians(config.maximum_turn_rate_deg_s),
                        max(-math.radians(config.maximum_turn_rate_deg_s), difference/dt))
        regime_override = 2
    else:
        regime_override = None
    new_speed = min(config.maximum_speed_m_s, max(config.minimum_speed_m_s, speed+acceleration*dt))
    maximum_turn_rate = min(math.radians(config.maximum_turn_rate_deg_s),
                           config.maximum_lateral_acceleration_m_s2/max(speed, new_speed, 1e-8))
    turn_rate = min(maximum_turn_rate, max(-maximum_turn_rate, turn_rate))
    new_heading = heading+turn_rate*dt
    new_velocity = np.array([new_speed*math.cos(new_heading), new_speed*math.sin(new_heading), 0.])
    return position+(velocity+new_velocity)*(.5*dt), new_velocity, regime_override


def _future(initial_position, initial_velocity, steps, dt, rng, cohort, config):
    position, velocity = np.empty((steps+1, 3)), np.empty((steps+1, 3))
    regime = np.zeros(steps+1, dtype=np.int8)
    onset = np.zeros(steps+1, dtype=bool)
    event_id = np.zeros(steps+1, dtype=np.int32)
    position[0], velocity[0], regime[0] = initial_position, initial_velocity, cohort if cohort != 4 else 0
    end, event_index, actual, acceleration, turn_rate = 0, -1, -1, 0., 0.
    for t in range(1, steps+1):
        if t >= end:
            event_index += 1
            actual, acceleration, turn_rate = _controls(rng, cohort, actual, event_index, config)
            duration = rng.uniform(config.minimum_event_duration_s, config.maximum_event_duration_s)
            end = t+max(1, int(math.ceil(duration/dt)))
            onset[t] = cohort != 0
        position[t], velocity[t], override = _advance(position[t-1], velocity[t-1], acceleration, turn_rate, dt, config)
        regime[t] = actual if override is None else override
        if override is not None and regime[t-1] != override:
            onset[t] = True
        event_id[t] = event_index
    return {'position': position.astype(np.float32), 'velocity': velocity.astype(np.float32),
            'beam': centered_dft_beams(position), 'motion_type': regime,
            'maneuver_onset': onset, 'regime_event_id': event_id}


def generate_split(count, steps=1001, dt=.05, seed=20261120, id_offset=0,
                   kind=None, initial_references=None, config=None):
    """Generate an independent split; callers must use disjoint seeds and IDs.

    With kind=None, trajectory ID modulo five balances the five cohorts.
    ``motion_type`` labels the current component (0..3); ``trajectory_kind``
    identifies the complete cohort (0..4), including mixed trajectories.
    Reference anchors must come from training data only. They constrain geometry
    but no recorded future trajectory is copied. Altitude is constant per track.
    """
    config = (config or ManeuverConfig()).validate()
    if (not isinstance(count, (int, np.integer)) or count < 1 or
            not isinstance(steps, (int, np.integer)) or steps < 2 or
            not isinstance(id_offset, (int, np.integer)) or id_offset < 0 or
            not isinstance(seed, (int, np.integer)) or seed < 0 or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Count, steps, seed, ID offset and dt must be valid')
    positions, velocities = _reference_pool(initial_references)
    ids = np.arange(id_offset, id_offset+count, dtype=np.int64)
    rows, kinds = [], []
    for identity in ids:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(identity)]))
        cohort = int(identity % 5) if kind is None else _kind_id(kind)
        initial = _initial_state(rng, positions, velocities, config)
        rows.append(_future(*initial, steps-1, dt, rng, cohort, config))
        kinds.append(cohort)
    result = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    result.update(trajectory_id=ids, trajectory_kind=np.asarray(kinds, dtype=np.int8))
    return result


def fork_future(prefix_position, prefix_velocity, steps=20, dt=.05, seed=20261120,
                kind='mixed', config=None):
    """Append ``steps`` new points to an unchanged measured-history reference.

    Different seeds share exactly the supplied position/velocity prefix but have
    independent future events. This is a simulator counterfactual, not a claim
    that an unobserved random future can be identified from the shared history.
    """
    config = (config or ManeuverConfig()).validate()
    position, velocity = np.asarray(prefix_position), np.asarray(prefix_velocity)
    if (position.ndim != 2 or position.shape[-1] != 3 or position.shape != velocity.shape or
            len(position) < 1 or not np.isfinite(position).all() or not np.isfinite(velocity).all() or
            not isinstance(steps, (int, np.integer)) or steps < 1 or not np.isfinite(dt) or dt <= 0):
        raise ValueError('Provide finite matching [history,3] position/velocity and a valid future length')
    cohort = _kind_id(kind)
    rng = np.random.default_rng(seed)
    future = _future(position[-1], velocity[-1], steps, dt, rng, cohort, config)
    total_position = np.concatenate((position, future['position'][1:]), axis=0)
    total_velocity = np.concatenate((velocity, future['velocity'][1:]), axis=0)
    result = {'position': total_position, 'velocity': total_velocity,
              'beam': centered_dft_beams(total_position),
              'motion_type': np.r_[np.full(len(position), -1, dtype=np.int8), future['motion_type'][1:]],
              'maneuver_onset': np.r_[np.zeros(len(position), bool), future['maneuver_onset'][1:]],
              'fork_index': len(position), 'trajectory_kind': cohort}
    return result


def protocol(config=None):
    return {'motion_config': asdict((config or ManeuverConfig()).validate()),
            'motion_names': MOTION_NAMES, 'altitude': 'constant_per_trajectory',
            'integration': 'trapezoidal_position_continuous_velocity',
            'tangential_acceleration_is_not_total_vector_acceleration': True,
            'effective_turn_rate_rad_s': 'min(command_limit_rad_s, maximum_lateral_acceleration/max(old_speed,new_speed))',
            'future_event_schedule_and_labels_are_evaluation_only': True,
            'anchor_usage': 'initial_geometry_only_from_training_reference_pool; no_future_path_copied',
            'steering': 'smooth_heading_control_near_configured_radius_without_coordinate_clipping; turn_labels_and_onsets_recorded'}
