"""
Actor-critic networks for the room/terrain policies, shared by training (train_fault.py) and every
script that loads a trained policy (fault_rollout.load_policy).

encoder="mlp": the original Brax MLP over the whole observation (camera pixels included as a flat
vector). Used by every run before 2026-11-20.

encoder="cnn": the actor's observation ends with the body-camera frames (room_env.CAM_FRAMES, each
CAM_H x CAM_W, see room_env._get_obs). A small conv net turns them into a 128-number feature, which
is joined to the rest of the observation (IMU, last commands, path features) before the MLP. The
conv net sees the newest frame plus its differences to the two older ones, so motion (optical flow)
is easy to pick up. The critic is unchanged: it reads the privileged state, which has no image.
"""
import functools
from typing import Sequence

import jax.numpy as jp
from brax.training import networks
from brax.training.agents.ppo import networks as ppo_networks
from flax import linen

CONV = ((16, 5, 2), (32, 3, 2), (32, 3, 2), (32, 3, 1))     # (channels, kernel, stride)
CAM_FEAT = 128


class CamActor(linen.Module):
  n_frames: int
  h: int
  w: int
  hidden: Sequence[int]
  out: int

  @linen.compact
  def __call__(self, x):
    n_cam = self.n_frames * self.h * self.w
    rest, cam = x[..., :-n_cam], x[..., -n_cam:]
    lead = cam.shape[:-1]
    img = cam.reshape((-1, self.n_frames, self.h, self.w))
    img = jp.concatenate([img[:, :1], img[:, :1] - img[:, 1:]], 1)        # newest frame, motion
    img = jp.moveaxis(img, 1, -1)                                         # (B, H, W, C)
    for c, k, s in CONV:
      img = linen.relu(linen.Conv(c, (k, k), strides=(s, s))(img))
    z = linen.relu(linen.Dense(CAM_FEAT)(img.reshape(img.shape[0], -1)))
    h = jp.concatenate([rest, z.reshape(lead + (CAM_FEAT,))], -1)
    for n in self.hidden:
      h = linen.swish(linen.Dense(n)(h))
    return linen.Dense(self.out)(h)


def make(observation_size, action_size, preprocess_observations_fn, policy_hidden_layer_sizes,
         value_hidden_layer_sizes, encoder="mlp", cam=None, **_):
  """Brax PPO networks; cam = (n_frames, h, w) when the observation has camera frames."""
  nets = ppo_networks.make_ppo_networks(
      observation_size, action_size, preprocess_observations_fn=preprocess_observations_fn,
      policy_hidden_layer_sizes=policy_hidden_layer_sizes, value_hidden_layer_sizes=value_hidden_layer_sizes,
      policy_obs_key="state", value_obs_key="privileged_state")
  if encoder != "cnn" or cam is None:
    return nets
  module = CamActor(*cam, hidden=tuple(policy_hidden_layer_sizes),
                    out=nets.parametric_action_distribution.param_size)
  size = observation_size["state"][0]

  def apply(processor_params, policy_params, obs):
    x = preprocess_observations_fn(obs["state"], networks.normalizer_select(processor_params, "state"))
    return module.apply(policy_params, x)

  policy = networks.FeedForwardNetwork(init=lambda key: module.init(key, jp.zeros((1, size))), apply=apply)
  return nets.replace(policy_network=policy) if hasattr(nets, "replace") else \
      ppo_networks.PPONetworks(policy_network=policy, value_network=nets.value_network,
                               parametric_action_distribution=nets.parametric_action_distribution)


def factory(encoder, cam, policy_layers, value_layers):
  return functools.partial(make, policy_hidden_layer_sizes=policy_layers, value_hidden_layer_sizes=value_layers,
                           encoder=encoder, cam=cam)
