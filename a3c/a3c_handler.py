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

from a3c.run_loop import run_loop

COUNTER = 0
LOCK = threading.Lock()

FLAGS = flags.FLAGS

if FLAGS.training:
  PARALLEL = FLAGS.parallel
  MAX_AGENT_STEPS = FLAGS.max_agent_steps
  DEVICE = ['/cpu:0']
  #DEVICE = ['/gpu:'+dev for dev in FLAGS.device.split(',')]
else:
  PARALLEL = 1
  MAX_AGENT_STEPS = 1e5
  DEVICE = ['/cpu:0']


LOG = FLAGS.log_path+FLAGS.map+'/'+FLAGS.net
SNAPSHOT = FLAGS.snapshot_path+FLAGS.map+'/'+FLAGS.net
if not os.path.exists(LOG):
  os.makedirs(LOG)
if not os.path.exists(SNAPSHOT):
  os.makedirs(SNAPSHOT)

def run_thread(agent, map_name, visualize):
  with sc2_env.SC2Env(
    map_name=map_name,
    step_mul=FLAGS.step_mul,
    agent_interface_format=AgentInterfaceFormat(
        feature_dimensions = Dimensions((FLAGS.screen_resolution,FLAGS.screen_resolution),(FLAGS.minimap_resolution,FLAGS.minimap_resolution)),
        action_space = ActionSpace.FEATURES,
        camera_width_world_units = 24),
    visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)

    # Only for a single player!
    replay_buffer = []
    for recorder, is_done in run_loop([agent], env, MAX_AGENT_STEPS):
      if FLAGS.training:
        replay_buffer.append(recorder)
        if is_done:
          counter = 0
          with LOCK:
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


def a3c_handler(unused_argv):
  """Run agents"""
  stopwatch.sw.enabled = FLAGS.profile or FLAGS.trace
  stopwatch.sw.trace = FLAGS.trace

  maps.get(FLAGS.map)  # Assert the map exists.

  # Setup agents
  agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
  agent_cls = getattr(importlib.import_module(agent_module), agent_name)

  agents = []
  for i in range(PARALLEL):
    agent = agent_cls(FLAGS.training, FLAGS.minimap_resolution, FLAGS.screen_resolution)
    agent.build_model(i > 0, DEVICE[i % len(DEVICE)], FLAGS.net)
    agents.append(agent)

  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  summary_writer = tf.summary.FileWriter(LOG)
  for i in range(PARALLEL):
    agents[i].setup(sess, summary_writer)

  agent.initialize()
  if not FLAGS.training or FLAGS.continuation:
    global COUNTER
    COUNTER = agent.load_model(SNAPSHOT)

  # Run threads
  threads = []
  for i in range(PARALLEL - 1):
    t = threading.Thread(target=run_thread, args=(agents[i], FLAGS.map, False))
    threads.append(t)
    t.daemon = True
    t.start()
    time.sleep(5)

  run_thread(agents[-1], FLAGS.map, FLAGS.render)

  for t in threads:
    t.join()

  if FLAGS.profile:
    print(stopwatch.sw)
