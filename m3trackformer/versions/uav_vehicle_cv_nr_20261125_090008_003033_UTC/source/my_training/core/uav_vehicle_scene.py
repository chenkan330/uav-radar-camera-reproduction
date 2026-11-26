"""Preplanned level UAV flight and a straight, constant-speed ground vehicle in ENU metres."""

from dataclasses import dataclass

import numpy as np


DT = 0.05
SUBSTEPS = 10
ALTITUDE_M = 60.0
VEHICLE_SPEED_MPS = 80.0 / 3.6
INITIAL_HORIZONTAL_RANGE_M = (280.0, 510.0)
INITIAL_BEARING_FROM_NORTH_DEG = (-18.0, 18.0)
ROAD_ANGLE_FROM_EAST_WEST_DEG = (-25.0, 25.0)


@dataclass
class VehicleScene:
    time: np.ndarray
    platform: np.ndarray
    vehicle: np.ndarray
    command: np.ndarray
    noise: np.ndarray
    speed: np.ndarray
    turn_rate: np.ndarray


def build_scene(seed, frames=200):
    """Generate synchronized truth, flight-plan increments and standard-normal noise.

    Aircraft velocity may turn while the array axes remain fixed in ENU.
    command[k] is the preplanned ENU displacement from frame k to frame k+1.
    The vehicle and flight plan are sampled independently.
    """
    if frames < 2:
        raise ValueError("A scene needs at least two frames.")
    flight_seed, road_seed, noise_seed = np.random.SeedSequence(seed).spawn(3)
    flight = np.random.default_rng(flight_seed)
    road = np.random.default_rng(road_seed)
    time = DT * np.arange(frames)

    speed_mean = flight.uniform(11, 14)
    speed_amplitude = flight.uniform([0.8, 0.3], [1.8, 0.7])
    speed_frequency = 2 * np.pi / flight.uniform([7, 4], [13, 7])
    speed_phase = flight.uniform(-np.pi, np.pi, 2)
    initial_course = flight.uniform(-np.pi, np.pi)

    # Include broad, one-direction turns as well as S turns with curvature reversals.
    sustained_turn = flight.random() < 0.5
    bias_range, swing_range = ((8, 12), (3, 6)) if sustained_turn else ((1, 3), (11, 16))
    turn_bias = np.deg2rad(flight.choice([-1, 1]) * flight.uniform(*bias_range))
    turn_amplitude = np.deg2rad([flight.uniform(*swing_range), flight.uniform(0.8, 2.5)])
    turn_frequency = 2 * np.pi / flight.uniform([6, 4], [10, 7])
    turn_phase = flight.uniform(-np.pi, np.pi, 2)

    def speed_at(t):
        return speed_mean + np.sin(t[:, None] * speed_frequency + speed_phase) @ speed_amplitude

    # Heading is the analytic integral of the smooth turn-rate commands.
    # Integrate velocity on 5 ms midpoints before applying any simulated position updates.
    step_dt = DT / SUBSTEPS
    midpoints = (np.arange((frames - 1) * SUBSTEPS) + 0.5) * step_dt
    turn_integrals = (np.cos(turn_phase) - np.cos(midpoints[:, None] * turn_frequency + turn_phase))
    course = initial_course + turn_bias * midpoints + turn_integrals @ (turn_amplitude / turn_frequency)
    velocity_xy = speed_at(midpoints)[:, None] * np.column_stack((np.cos(course), np.sin(course)))
    command = np.zeros((frames - 1, 3))
    command[:, :2] = step_dt * velocity_xy.reshape(frames - 1, SUBSTEPS, 2).sum(axis=1)
    platform = np.empty((frames, 3))
    platform[0] = [0, 0, ALTITUDE_M]
    platform[1:] = platform[0] + np.cumsum(command, axis=0)

    # Initial bearing is measured clockwise from north; road heading is measured from east.
    horizontal_range = road.uniform(*INITIAL_HORIZONTAL_RANGE_M)
    bearing = np.deg2rad(road.uniform(*INITIAL_BEARING_FROM_NORTH_DEG))
    road_point = horizontal_range * np.array([np.sin(bearing), np.cos(bearing), 0])
    road_heading = road.choice([0, np.pi]) + np.deg2rad(road.uniform(*ROAD_ANGLE_FROM_EAST_WEST_DEG))
    vehicle_velocity = VEHICLE_SPEED_MPS * np.array([np.cos(road_heading), np.sin(road_heading), 0])
    vehicle = road_point + time[:, None] * vehicle_velocity
    noise = np.random.default_rng(noise_seed).standard_normal((frames, 3))
    turn_rate = turn_bias + np.sin(time[:, None] * turn_frequency + turn_phase) @ turn_amplitude
    return VehicleScene(time, platform, vehicle, command, noise, speed_at(time), turn_rate)
