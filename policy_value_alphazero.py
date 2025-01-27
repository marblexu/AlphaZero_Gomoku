# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Tested in PyTorch 0.2.0 and 0.3.0

@author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def sequential_pack(layers):
    """
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`List[nn.Module]`): The input list of layers.
    Returns:
        - seq (:obj:`nn.Sequential`): Packed sequential container.
    """
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq

def conv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        pad_type: str = 'zero',
        activation: nn.Module = None,
        norm_type: str = None,
        num_groups_for_gn: int = 1,
        bias: bool = True
) -> nn.Sequential:
    block = []
    block.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    )
    block.append(nn.BatchNorm2d(out_channels))
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)

def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn = None,
    activation: nn.Module = None,
    norm_type: str = None,
    dropout_probability: float = 0.5,
    output_activation: bool = True,
    output_norm: bool = True,
    last_linear_layer_init_zero: bool = False
):
    """
    Overview:
        Create a multi-layer perceptron using fully-connected blocks with activation, normalization, and dropout,
        optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - layer_num (:obj:`int`): Number of layers.
        - layer_fn (:obj:`Callable`, optional): Layer function.
        - activation (:obj:`nn.Module`, optional): The optional activation function.
        - norm_type (:obj:`str`, optional): The type of the normalization.
        - use_dropout (:obj:`bool`, optional): Whether to use dropout in the fully-connected block. Default is False.
        - dropout_probability (:obj:`float`, optional): Probability of an element to be zeroed in the dropout. \
            Default is 0.5.
        - output_activation (:obj:`bool`, optional): Whether to use activation in the output layer. If True, \
            we use the same activation as front layers. Default is True.
        - output_norm (:obj:`bool`, optional): Whether to use normalization in the output layer. If True, \
            we use the same normalization as front layers. Default is True.
        - last_linear_layer_init_zero (:obj:`bool`, optional): Whether to use zero initializations for the last \
            linear layer (including w and b), which can provide stable zero outputs in the beginning, \
            usually used in the policy network in RL settings.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the multi-layer perceptron.

    .. note::
        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html).
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(nn.LayerNorm(out_channels))
        if activation is not None:
            block.append(activation)

    # The last layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))
    """
    In the final layer of a neural network, whether to use normalization and activation are typically determined
    based on user specifications. These specifications depend on the problem at hand and the desired properties of
    the model's output.
    """
    if output_norm is True:
        # The last layer uses the same norm as front layers.
        if norm_type is not None:
            block.append(nn.LayerNorm(out_channels))
    if output_activation is True:
        # The last layer uses the same activation as front layers.
        if activation is not None:
            block.append(activation)

    if last_linear_layer_init_zero:
        # Locate the last linear layer and initialize its weights and biases to 0.
        for _, layer in enumerate(reversed(block)):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return sequential_pack(block)
   
class ResBlock(nn.Module):
    """
    Overview:
        Residual Block with 2D convolution layers, including 3 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
    """
    def __init__(
        self,
        in_channels: int,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = 'BN',
        res_type: str = 'basic',
        bias: bool = True,
        out_channels = None,
    ) -> None:
        super(ResBlock, self).__init__()       
        self.act = activation
        if out_channels is None:
            out_channels = in_channels
        self.conv1 = conv2d_block(in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias)
        self.conv2 = conv2d_block(out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x + identity)
        return x
        
class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        action_space_size = int(1 * board_width * board_height)
        observation_shape = (4, board_width, board_height)
        num_channels = 32
        num_res_blocks = 1
        value_head_channels = 16
        policy_head_channels = 16
        value_head_hidden_channels = [32]
        policy_head_hidden_channels = [32]
        self.flatten_input_size_for_value_head = (value_head_channels * observation_shape[1] * observation_shape[2])
        self.flatten_input_size_for_policy_head = (policy_head_channels * observation_shape[1] * observation_shape[2])
        last_linear_layer_init_zero = True

        self.board_width = board_width
        self.board_height = board_height
        activation = nn.ReLU(inplace=True)
        # Representation network
        self.conv = nn.Conv2d(observation_shape[0], num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(num_channels)
        self.activation = activation
        self.resblock = ResBlock(in_channels=num_channels, activation=activation, norm_type='BN', bias=False)
        
        # Prediction network
        # Policy layers
        self.conv1x1_policy = nn.Conv2d(num_channels, policy_head_channels, kernel_size=1)
        self.norm_policy = nn.BatchNorm2d(policy_head_channels)
        self.fc_policy_head = MLP(
            in_channels=self.flatten_input_size_for_policy_head,
            hidden_channels=policy_head_hidden_channels[0],
            out_channels=action_space_size,
            layer_num=len(policy_head_hidden_channels) + 1,
            activation=activation,
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )
        # Value layers
        self.conv1x1_value = nn.Conv2d(num_channels, value_head_channels, kernel_size=1)
        self.norm_value = nn.BatchNorm2d(value_head_channels)
        self.fc_value_head = MLP(
            in_channels=self.flatten_input_size_for_value_head,
            hidden_channels=value_head_hidden_channels[0],
            out_channels=1,
            layer_num=len(value_head_hidden_channels) + 1,
            activation=activation,
            norm_type='LN',
            output_activation=False,
            output_norm=False,
            last_linear_layer_init_zero=last_linear_layer_init_zero
        )

    def forward(self, x):
        # Representation network
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.resblock(x)
        
        # Prediction network
        x = self.resblock(x)
        # Policy layers
        policy = self.conv1x1_policy(x)
        policy = self.norm_policy(policy)
        policy = self.activation(policy)
        policy = policy.reshape(-1, self.flatten_input_size_for_policy_head)
        policy = self.fc_policy_head(policy)
        policy = F.log_softmax(policy, dim=-1)       
        # Value layers
        value = self.conv1x1_value(x)
        value = self.norm_value(value)
        value = self.activation(value)
        value = value.reshape(-1, self.flatten_input_size_for_value_head)
        value = self.fc_value_head(value)
        return policy, value


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=False):
        self.use_gpu = torch.cuda.is_available()
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height))
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
            value = value.data.cpu().numpy()[0][0]
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
            value = value.data.numpy()[0][0]
        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).cuda())
            winner_batch = Variable(torch.FloatTensor(np.array(winner_batch)).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
