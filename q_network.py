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


class DQNOutputNDOp(mx.operator.NDArrayOp):

    def __init__(self):
        super(DQNOutputNDOp, self).__init__(need_top_grad=False)
        self.fwd_kernel = None
        self.bwd_kernel = None

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1]
        target = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        if self.bwd_kernel is None:
            self.bwd_kernel = mx.rtc(
                'dqn', [('x', x), ('action', action), ('target', target)],
                [('dx', dx)], """
            int i = blockIdx.x;
            int j = threadIdx.x;
            int k = static_cast<int>(action[i]);
            float tmp;
            if(j == k){
                tmp = x[i*x_dims[1]+j] - target[i*x_dims[1]];
                if(tmp > 1.0f)
                    tmp = 1.0f;
                if(tmp < -1.0f)
                    tmp = -1.0f;
                dx[i*x_dims[1]+j] = tmp;
            }
            """)
        self.bwd_kernel.push([x, action, target], [dx], (x.shape[0], 1, 1),
                             (x.shape[1], 1, 1))

        # dx[np.arange(action.shape[0]), action] \
        #    = np.clip(x[np.arange(action.shape[0]), action] - reward, -1, 1)


class DQNOutputNpyOp(mx.operator.NumpyOp):

    def __init__(self):
        super(DQNOutputNpyOp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'action', 'reward']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        action_shape = (in_shape[0][0],)
        reward_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, action_shape, reward_shape], [output_shape]

    def forward(self, in_data, out_data):
        x = in_data[0]
        y = out_data[0]
        y[:] = x

    def backward(self, out_grad, in_data, out_data, in_grad):
        x = out_data[0]
        action = in_data[1].astype(np.int)
        reward = in_data[2]
        dx = in_grad[0]
        dx[:] = 0
        dx[np.arange(action.shape[0]), action] \
            = np.clip(x[np.arange(action.shape[0]), action] - reward, -1, 1)


class DQNInitializer(mx.initializer.Xavier):

    def _init_bias(self, _, arr):
        arr[:] = .1

    def _init_default(self, name, _):
        pass


class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self,
                 input_width,
                 input_height,
                 num_actions,
                 num_frames,
                 discount,
                 learning_rate,
                 rho,
                 rms_epsilon,
                 momentum,
                 clip_delta,
                 freeze_interval,
                 batch_size,
                 network_type,
                 update_rule,
                 batch_accumulator,
                 rng,
                 double=False,
                 input_scale=255.0,
                 ctx=mx.gpu(0)):

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
        self.double = double

        input_shape = (batch_size, num_frames, input_width, input_height)
        net = self.build_nature_network(num_actions)

        self.loss_exe = net.simple_bind(
            ctx=ctx, grad_req='write', data=input_shape)
        self.target_exe = net.simple_bind(
            ctx=ctx, grad_req='null', data=input_shape)
        self.policy_exe = self.loss_exe.reshape(
            data=(1, num_frames, input_width, input_height),
            partial_shaping=True)

        initializer = DQNInitializer(factor_type='in')
        names = net.list_arguments()
        for name in names:
            initializer(name, self.loss_exe.arg_dict[name])

        self.target_exe.copy_params_from(arg_params=self.loss_exe.arg_dict)
        self.optimizer = mx.optimizer.create(
            name='adagrad',
            learning_rate=0.01,
            eps=0.01,
            wd=0.0,
            clip_gradient=None,
            rescale_grad=1.0)

        '''
        lr_scheduler = mx.lr_scheduler.FactorScheduler(step=50000, factor=0.96)
        self.optimizer = mx.optimizer.create(
            name='rmsprop',
            learning_rate=0.00025,
            epsilon=0.01,  # Small value to avoid divide by zero
            gamma2=0.95,  # Moving average decay factor
            gamma1=0.9,  # Momentum factor
            lr_scheduler=lr_scheduler)
        '''
        self.updater = mx.optimizer.get_updater(self.optimizer)

    @staticmethod
    def update_weights(executor, updater):
        for ind, k in enumerate(executor.arg_dict):
            if k.endswith('weight') or k.endswith('bias'):
                updater(
                    index=ind,
                    grad=executor.grad_dict[k],
                    weight=executor.arg_dict[k])

    def load_weights(self, params):
        self.policy_exe.copy_params_from(params)
        self.copy_weights(self.policy_exe, self.loss_exe)
        self.copy_weights(self.policy_exe, self.policy_exe)

    @staticmethod
    def build_nature_network(num_actions=20):
        data = mx.sym.Variable("data")
        conv1 = mx.sym.Convolution(
            data=data,
            num_filter=32,
            stride=(4, 4),
            kernel=(8, 8),
            name="conv1")
        relu1 = mx.sym.Activation(data=conv1, act_type='relu', name="relu1")
        conv2 = mx.sym.Convolution(
            data=relu1,
            num_filter=64,
            stride=(2, 2),
            kernel=(4, 4),
            name="conv2")
        relu2 = mx.sym.Activation(data=conv2, act_type='relu', name="relu2")
        conv3 = mx.sym.Convolution(
            data=relu2,
            num_filter=64,
            stride=(1, 1),
            kernel=(3, 3),
            name="conv3")
        relu3 = mx.sym.Activation(data=conv3, act_type='relu', name="relu3")
        fc4 = mx.sym.FullyConnected(data=relu3, name="fc4", num_hidden=512)
        relu4 = mx.sym.Activation(data=fc4, act_type='relu', name="relu4")
        fc5 = mx.sym.FullyConnected(
            data=relu4, name="fc5", num_hidden=num_actions)
        # dqn = DQNOutputNDOp()
        dqn = DQNOutputNpyOp()
        out = dqn(data=fc5, name='dqn')
        return out

    def train(self, imgs, actions, rewards, terminals, R):
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
        rt = mx.nd.array(rewards[:, 0], ctx=self.ctx)
        tt = mx.nd.array(terminals[:, 0], ctx=self.ctx)
        st1 = mx.nd.array(next_states, ctx=self.ctx) / self.input_scale
        R = mx.nd.array(R[:, 0], ctx=self.ctx)

        next_q_out = self.target_exe.forward(is_train=False, data=st1)[0]
        if self.double:
            next_q_value = self.loss_exe.forward(is_train=False, data=st1)[0]
            mx.nd.waitall()
            next_q_index = mx.nd.argmax_channel(next_q_value)
            next_q_out_ = mx.nd.choose_element_0index(next_q_out, next_q_index)
        else:
            next_q_out_ = mx.nd.max(next_q_out, axis=1)

        target_q_values = rt + next_q_out_ * (1.0 - tt) * self.discount

        current_q_out = self.loss_exe.forward(
            is_train=True, data=st, dqn_reward=target_q_values,
            dqn_action=at)[0]
        current_q_values = mx.nd.choose_element_0index(current_q_out, at)
        diff = mx.nd.clip(current_q_values - target_q_values, -1.0, 1.0)
        self.loss_exe.backward()
        self.update_weights(self.loss_exe, self.updater)

        if (self.freeze_interval > 0 and self.update_counter > 0 and
                self.update_counter % self.freeze_interval == 0):
            self.target_exe.copy_params_from(arg_params=self.loss_exe.arg_dict)

        self.update_counter += 1
        return mx.nd.sum(mx.nd.abs(diff)).asnumpy()

    def q_vals(self, state):
        st = mx.nd.array([state], ctx=self.ctx) / self.input_scale
        return self.policy_exe.forward(data=st)[0].asnumpy()

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.num_actions), 0
        q_vals = self.q_vals(state)
        return np.argmax(q_vals), np.max(q_vals)


def main():
    pass


if __name__ == '__main__':
    main()
