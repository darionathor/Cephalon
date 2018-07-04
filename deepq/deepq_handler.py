from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import threading
import time

import tensorflow as tf
from absl import flags
from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.env.sc2_env import AgentInterfaceFormat, Dimensions, ActionSpace
from pysc2.lib import stopwatch

from deepq.run_loop import run_loop

COUNTER = 0

FLAGS = flags.FLAGS

if FLAGS.training:
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/cpu:0']
  #DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']


LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net+'/'+FLAGS.agent
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net+'/'+FLAGS.agent
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)


def deepq_handler(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)

  agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
  agent.build_model(False, DEVICE[1 % len(DEVICE)], FLAGS.net)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True

  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  agent.setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  with sc2_env.SC2Env(
    map_name=FLAGS.map,
    step_mul=FLAGS.step_mul,
    agent_interface_format=AgentInterfaceFormat(
        feature_dimensions = Dimensions((FLAGS.screen_resolution,FLAGS.screen_resolution),(FLAGS.minimap_resolution,FLAGS.minimap_resolution)),
        action_space = ActionSpace.FEATURES,
        camera_width_world_units = FLAGS.camera_width),
    visualize=FLAGS.render) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)

    # Only for a single player!
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          global COUNTER
          COUNTER += 1
          counter = COUNTER
          # Learning rate schedule
          learning_rate = FLAGS.learning_rate * (1 - 0.9 * counter / FLAGS.max_steps)
          agent.update(replay_buffer, FLAGS.discount, learning_rate, counter)
          replay_buffer = []
          if counter % FLAGS.snapshot_step == 1:
            agent.save_model(SNAPSHOT, counter)
          if counter >= FLAGS.max_steps:
            break
      elif is_done:
        obs = recorder[-1].observation
        score = obs["score_cumulative"][0]
        print('Your score is '+str(score)+'!')
    if FLAGS.save_replay:
      env.save_replay(agent.name)

  if FLAGS.profile:
    print(stopwatch.sw)
