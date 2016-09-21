"""
Code for deep Q-learning as described in:

Playing Atari with Deep Reinforcement Learning
NIPS Deep Learning Workshop 2013

and

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015


Author of Lasagne port: Nissan Pow
Modifications: Nathan Sprague
"""

import numpy as np
import mxnet as mx


class DQNInitializer(mx.initializer.Uniform):
    def __init__(self):
        mx.initializer.Uniform.__init__(self)

    def _init_bias(self, _, arr):
        arr[:] = .1

    def _init_default(self, name, _):
        pass


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, input_width, input_height, num_actions,
                 num_frames, discount, learning_rate, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, input_scale=255.0, ctx=mx.gpu(0)):

        self.input_width = input_width
        self.input_height = input_height
        self.num_actions = num_actions
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        self.update_counter = 0
        self.ctx = ctx
        self.input_scale = input_scale

        input_shape = (batch_size, num_frames, input_width, input_height)
        self.loss_exe, self.policy_exe, self.target_exe = self.build_graphs(input_shape, ctx, num_actions)
        self.optimizer = mx.optimizer.create(name='rmsprop', learning_rate=self.lr, gamma2=self.momentum, clip_gradient=self.clip_delta)
        self.updater = mx.optimizer.get_updater(self.optimizer)
        initializer = DQNInitializer()
        self.init_exe(self.loss_exe, initializer)
        self.copy_weights(self.loss_exe, self.target_exe)
        self.copy_weights(self.loss_exe, self.policy_exe)

    @staticmethod
    def copy_weights(from_exe, to_exe):
        for k in from_exe.arg_dict:
            if k.endswith('weight') or k.endswith('bias'):
                from_exe.arg_dict[k].copyto(to_exe.arg_dict[k])

    @staticmethod
    def share_weights(source_exe, to_exe):
        for k in source_exe.arg_dict:
            if k.endswith('weight') or k.endswith('bias'):
                to_exe.arg_dict[k] = source_exe.arg_dict[k]

    @staticmethod
    def init_exe(executor, initializer):
        for k, v in executor.arg_dict.items():
            initializer(k, v)

    @staticmethod
    def update_weights(executor, updater):
        for ind, k in enumerate(executor.arg_dict):
            if k.endswith('weight') or k.endswith('bias'):
                updater(index=ind, grad=executor.grad_dict[k], weight=executor.arg_dict[k])

    def load_weights(self, params):
        self.policy_exe.copy_params_from(params)
        self.copy_weights(self.policy_exe, self.loss_exe)
        self.copy_weights(self.policy_exe, self.policy_exe)

    @staticmethod
    def build_nature_network(num_actions=20):
        data = mx.sym.Variable("data")
        conv1 = mx.sym.Convolution(data=data, num_filter=32, stride=(4, 4),
                                   kernel=(8, 8), name="conv1")
        relu1 = mx.sym.Activation(data=conv1, act_type='relu', name="relu1")
        conv2 = mx.sym.Convolution(data=relu1, num_filter=64, stride=(2, 2),
                                   kernel=(4, 4), name="conv2")
        relu2 = mx.sym.Activation(data=conv2, act_type='relu', name="relu2")
        conv3 = mx.sym.Convolution(data=relu2, num_filter=64, stride=(1, 1),
                                   kernel=(3, 3), name="conv3")
        relu3 = mx.sym.Activation(data=conv3, act_type='relu', name="relu3")
        fc4 = mx.sym.FullyConnected(data=relu3, name="fc4", num_hidden=512)
        relu4 = mx.sym.Activation(data=fc4, act_type='relu', name="relu4")
        fc5 = mx.sym.FullyConnected(data=relu4, name="fc5", num_hidden=num_actions)
        return fc5

    def build_graphs(self, input_shape, ctx, num_actions=20):
        batch_size, num_frames, input_width, input_height = input_shape
        q_values = self.build_nature_network(num_actions)
        target_q_values = mx.sym.Variable("target")
        action_mask = mx.sym.Variable("action")
        out_q_values = mx.sym.sum(q_values * action_mask, axis=1)
        loss = mx.sym.LinearRegressionOutput(data=out_q_values, label=target_q_values)
        loss_exe = loss.simple_bind(ctx=ctx, data=input_shape, grad_req='write')
        policy_exe = q_values.simple_bind(ctx=ctx, data=(1, num_frames, input_width, input_height), grad_req='null')
        target_exe = q_values.simple_bind(ctx=ctx, data=input_shape, grad_req='null')
        return loss_exe, policy_exe, target_exe

    def train(self, imgs, actions, rewards, terminals):
        """
        Train one batch.

        Arguments:

        imgs - b x (f + 1) x h x w numpy array, where b is batch size,
               f is num frames, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array
        terminals - b x 1 numpy boolean array (currently ignored)

        Returns: average loss
        """
        states = imgs[:, :-1, :, :]
        next_states = imgs[:, 1:, :, :]
        st = mx.nd.array(states, ctx=self.ctx) / self.input_scale
        at = mx.nd.array(actions[:, 0], ctx=self.ctx)
        at_encoded = mx.nd.zeros((self.batch_size, self.num_actions), ctx=self.ctx)
        mx.nd.onehot_encode(at, at_encoded)
        rt = mx.nd.array(rewards[:, 0], ctx=self.ctx)
        tt = mx.nd.array(terminals[:, 0], ctx=self.ctx)
        st1 = mx.nd.array(next_states, ctx=self.ctx) / self.input_scale

        next_q_values = self.target_exe.forward(data=st1)[0]
        target_q_values = rt + mx.nd.choose_element_0index(next_q_values, mx.nd.argmax_channel(next_q_values)) * (1.0 - tt) * self.discount
        out_q_values = self.loss_exe.forward(is_train=True, data=st, target=target_q_values, action=at_encoded)[0]
        self.loss_exe.backward()
        self.update_weights(self.loss_exe, self.updater)

        loss = mx.nd.square(out_q_values - target_q_values)
        loss = mx.nd.sum(loss)/self.batch_size
        self.copy_weights(self.loss_exe, self.policy_exe)

        if self.freeze_interval > 0 and self.update_counter % self.freeze_interval == 0:
            self.copy_weights(self.loss_exe, self.target_exe)
        self.update_counter += 1
        return loss.asnumpy()

    def q_vals(self, state):
        st = mx.nd.array([state], ctx=self.ctx) / self.input_scale
        return self.policy_exe.forward(data=st)[0].asnumpy()

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions)
        q_vals = self.q_vals(state)
        return np.argmax(q_vals)


def main():
    pass

if __name__ == '__main__':
    main()

