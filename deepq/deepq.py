from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
from pysc2.lib import actions

from deepq import utils as U
from deepq.network import build_net


class DeepQAgent(object):
  """An agent specifically for solving the mini-game maps."""
  def __init__(self, training, msize, ssize, name='deepq/DeepQAgent'):
    self.name = name
    self.training = training
    self.summary = []
    # Minimap size, screen size and info size
    assert msize == ssize
    self.msize = msize
    self.ssize = ssize
    self.isize = len(actions.FUNCTIONS)
    self.iter = 0


  def setup(self, sess, summary_writer):
    self.sess = sess
    self.summary_writer = summary_writer


  def initialize(self):
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)


  def reset(self):
    # Epsilon schedule
    self.epsilon = [0.05, 0.2]

  def copy_model_parameters(self):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: Tensorflow session instance
      estimator1: Estimator to copy the paramters from
      estimator2: Estimator to copy the parameters to
    """
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(self.name)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith('evaluator')]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
      op = e2_v.assign(e1_v)
      update_ops.append(op)

    self.sess.run(update_ops)

  def build_model(self, reuse, dev, ntype):
    with tf.variable_scope('evaluator') and tf.device(dev):
      self.minimapQ = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screenQ = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.infoQ = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = build_net(self.minimapQ, self.screenQ, self.infoQ, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
      # self.spatial_action, self.non_spatial_action, self.value = net
      self.spatial_Q, self.non_spatial_Q = net
    with tf.variable_scope(self.name) and tf.device(dev):
      if reuse:
        tf.get_variable_scope().reuse_variables()
        assert tf.get_variable_scope().reuse

      # Set inputs of networks
      self.minimap = tf.placeholder(tf.float32, [None, U.minimap_channel(), self.msize, self.msize], name='minimap')
      self.screen = tf.placeholder(tf.float32, [None, U.screen_channel(), self.ssize, self.ssize], name='screen')
      self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

      # Build networks
      net = build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS), ntype)
      #self.spatial_action, self.non_spatial_action, self.value = net
      self.spatial_action, self.non_spatial_action, self.q_value = net

      # Set targets and masks
      #self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
      #self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize**2], name='spatial_action_selected')
      #self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='valid_non_spatial_action')
      #self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name='non_spatial_action_selected')
      self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

      # Compute log probability
      #spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
      #spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
      #non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
      # valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
      # valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
      # non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
      # non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
      #self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
      #self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

      #value loss
      #action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
      #advantage = tf.stop_gradient(self.value_target - self.value)
      #policy_loss = - tf.reduce_mean(action_log_prob * advantage)
      #value_loss = - tf.reduce_mean(self.value * advantage)
      #self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
      #self.summary.append(tf.summary.scalar('value_loss', value_loss))


      # TODO: policy penalty
      #loss = value_loss + C:\Users\sasal\PycharmProjects\sc2_cephalon_loss
      #self.q_spatial = tf.reduce_mean(tf.square(self.spatial_action_selected - self.spatial_action));
      #self.q_non_spatial = tf.reduce_mean(tf.square(self.non_spatial_action_selected - self.non_spatial_action));
      self.loss = tf.reduce_mean(tf.squared_difference(self.value_target,  self.q_value));
      self.summary.append(tf.summary.scalar('value_loss', self.loss))
      # Build the optimizer
      self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
      opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
      # grads = opt.compute_gradients(loss)
      # cliped_grad = []
      # for grad, var in grads:
      #   self.summary.append(tf.summary.histogram(var.op.name, var))
      #   self.summary.append(tf.summary.histogram(var.op.name+'/grad', grad))
      #   grad = tf.clip_by_norm(grad, 10.0)
      #   cliped_grad.append([grad, var])
      self.train_op = opt.minimize(self.loss)#.apply_gradients(cliped_grad)
      self.summary_op = tf.summary.merge(self.summary)

      self.saver = tf.train.Saver(max_to_keep=100)


  def step(self, obs):
    minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
    minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
    screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    # TODO: only use available actions
    info = np.zeros([1, self.isize], dtype=np.float32)
    info[0, obs.observation['available_actions']] = 1

    feed = {self.minimap: minimap,
            self.screen: screen,
            self.info: info}
    non_spatial_action, spatial_action = self.sess.run(
      [self.non_spatial_action, self.spatial_action],
      feed_dict=feed)

    # Select an action and a spatial target
    non_spatial_action = non_spatial_action.ravel()
    spatial_action = spatial_action.ravel()
    valid_actions = obs.observation['available_actions']
    act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
    target = np.argmax(spatial_action)
    target = [int(target // self.ssize), int(target % self.ssize)]

    if False:
      print(actions.FUNCTIONS[act_id].name, target)

    # Epsilon greedy exploration
    if self.training and np.random.rand() < self.epsilon[0]:
      act_id = np.random.choice(valid_actions)
    if self.training and np.random.rand() < self.epsilon[1]:
      dy = np.random.randint(-4, 5)
      target[0] = int(max(0, min(self.ssize-1, target[0]+dy)))
      dx = np.random.randint(-4, 5)
      target[1] = int(max(0, min(self.ssize-1, target[1]+dx)))

    # Set act_id and act_args
    act_args = []
    for arg in actions.FUNCTIONS[act_id].args:
      if arg.name in ('screen', 'minimap', 'screen2'):
        act_args.append([target[1], target[0]])
      else:
        act_args.append([0])  # TODO: Be careful
    return actions.FunctionCall(act_id, act_args)


  def update(self, rbs, disc, lr, cter, batch_size):
    if(self.iter % 100 == 0):
      self.copy_model_parameters()
    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
    # batch_index = np.random.choice(len(rbs), size = 64, replace = False)
    # batch = [rbs[i] for i in batch_index]
    if(len(rbs)<batch_size):
      batch = rbs
    else:
      batch = random.sample(rbs,batch_size)
    # obs = batch[-1][-1]
    # if obs.last():
    #   Q = obs.reward
    # else:
    #   minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
    #   minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
    #   screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
    #   screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
    #   info = np.zeros([1, self.isize], dtype=np.float32)
    #   info[0, obs.observation['available_actions']] = 1
    #
    #   feed = {self.minimap: minimap,
    #           self.screen: screen,
    #           self.info: info}
    #   non_spatial_action_next_state, spatial_action_next_state = self.sess.run(
    #     [self.non_spatial_action, self.spatial_action],
    #     feed_dict=feed)
    #
    # # Compute targets and masks
    minimaps = []
    screens = []
    infos = []

    value_target = np.zeros([len(batch)], dtype=np.float32)
    non_spatial_value_target = np.zeros([len(batch)], dtype=np.float32)
    # value_target[-1] = Q

    #valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
    #spatial_action_selected = np.zeros([len(batch), self.ssize**2], dtype=np.float32)
    #valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
   # non_spatial_action_selected = np.zeros([len(batch), len(actions.FUNCTIONS)], dtype=np.float32)

    #rbs.reverse()
    # TODO Change to random batch selection?
    for i, [obs, action, next_obs] in enumerate(batch):
      minimap = np.array(obs.observation['feature_minimap'], dtype=np.float32)
      minimap = np.expand_dims(U.preprocess_minimap(minimap), axis=0)
      screen = np.array(obs.observation['feature_screen'], dtype=np.float32)
      screen = np.expand_dims(U.preprocess_screen(screen), axis=0)
      info = np.zeros([1, self.isize], dtype=np.float32)
      info[0, obs.observation['available_actions']] = 1
      feed = {self.minimap: minimap,
                        self.screen: screen,
                        self.info: info}
      minimaps.append(minimap)
      screens.append(screen)
      infos.append(info)

      non_spatial_action_next_state, spatial_action_next_state = self.sess.run(
        [self.non_spatial_Q, self.spatial_Q],
        feed_dict=feed)
     # non_spatial_action_next_state.ravel()
     # spatial_action_next_state.ravel()
      #q_value_next_state.ravel()

      reward = obs.reward
      act_id = action.function
      act_args = action.arguments


      #valid_actions = obs.observation["available_actions"]
      #valid_non_spatial_action[i, valid_actions] = 1
      #print(non_spatial_action_next_state)
      if(obs.last()):
       # non_spatial_action_selected[i, act_id] = 1
        value_target[i] = reward
      else:
       # non_spatial_action_selected[i, act_id] = 1
        value_target[i] = reward + disc * np.max(spatial_action_next_state[0])

      args = actions.FUNCTIONS[act_id].args
      for arg, act_arg in zip(args, act_args):
        if arg.name in ('screen', 'minimap', 'screen2'):
          ind = act_arg[1] * self.ssize + act_arg[0]
          #valid_spatial_action[i] = 1
          if(obs.last()):
              non_spatial_value_target[i] = reward
          else:
              non_spatial_value_target[i] =  reward + disc * np.max(non_spatial_action_next_state[0])

    minimaps = np.concatenate(minimaps, axis=0)
    screens = np.concatenate(screens, axis=0)
    infos = np.concatenate(infos, axis=0)

    # Train
    feed = {self.minimap: minimaps,
            self.screen: screens,
            self.info: infos,
            self.value_target: value_target,
            self.non_spatial_value_target: non_spatial_value_target,
            #self.valid_spatial_action: valid_spatial_action,
            #self.spatial_action_selected: spatial_action_selected,
            #self.valid_non_spatial_action: valid_non_spatial_action,
            #self.non_spatial_action_selected: non_spatial_action_selected,
            self.learning_rate: lr}
    _, summary, loss = self.sess.run([self.train_op, self.summary_op, self.loss], feed_dict=feed)
    print(loss)
    self.summary_writer.add_summary(summary, cter)
    self.iter = self.iter+1


  def save_model(self, path, count):
    self.saver.save(self.sess, path+'/model.pkl', count)


  def load_model(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    return int(ckpt.model_checkpoint_path.split('-')[-1])