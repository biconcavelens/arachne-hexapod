"""
Fourth environment stage: the rough outdoor field (terrain_env) plus things that move by themselves
across the path, as in the dynamic room (room_dyn_env): a dog, a rolling football, a hiker and a robot
lawnmower (terrain_scene.MOVERS). Each is present with probability p_mover and crosses the path once,
timed to meet the robot (3 s early to 3 s late); they follow the ground height. The dog, the hiker
and the mower stop while the robot is in their way; the football does not. Walking into a mover ends
the episode; being hit from the side or behind does not (see room_dyn_env). The critic also sees
their positions and velocities relative to the robot.
"""
import os

import jax.numpy as jp

import room_dyn_env as RD
import terrain_env as TE
import terrain_scene as TS

if not os.path.exists(TS.DYN_OUT):
    TS.build()


class FaultTerrainDyn(RD.FaultRoomDyn, TE.FaultTerrain):
  """MRO: mover logic (FaultRoomDyn) -> heightfield, start spot, camera march (FaultTerrain) -> FaultRoom."""
  XML = TS.DYN_OUT
  MOV_FOOT = jp.array([f for _, f, _ in TS.MOVERS])
  MOV_SPEED = jp.array(TS.MOV_SPEED)
  MOV_ROLL_R = jp.array(TS.MOV_ROLL_R)
  MOV_YIELD = jp.array([True, False, True, True])           # dog, football, hiker, mower
  MOV_LIM = TS.HALF - 0.1

  def __init__(self, config=None, config_overrides=None):
    super().__init__(config or RD.default_config(), config_overrides)
