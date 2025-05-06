"""
Defines the SAE model classes.
(Originally from https://github.com/ElanaPearl/InterPLM)
"""

import collections
import copy
from abc import ABC, abstractmethod

import torch as t
import torch.nn as nn
from utils.configs import get_logger

logger = get_logger("SAE-Model")


class Dictionary(ABC):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size, tied=True):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        if tied:
            dec_weight = self.encoder.weight.data.T.clone()
        else:
            dec_weight = t.randn_like(self.decoder.weight)

        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

        self.init_model_parameter = copy.deepcopy(self.state_dict())

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, mask=None, ghost_mask=None):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        mask: 0 for the features to be masked, 1 for other features
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:
            if mask is None:  # normal mode
                f = self.encode(x)
                x_hat = self.decode(f)
                if output_features:
                    return x_hat, f
                else:
                    return x_hat

            else:
                f = self.encode(x)
                f_mask = f * mask.to(f)
                x_hat = self.decode(f_mask)
                if output_features:
                    return x_hat, f_mask
                else:
                    return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(f_ghost)  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    @classmethod
    def from_pretrained(cls, checkpoint, para_name_map_dict={}, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        if isinstance(checkpoint, str):
            state_dict = t.load(checkpoint, map_location=t.device("cpu"))
        elif isinstance(checkpoint, collections.OrderedDict) or isinstance(checkpoint, dict):
            state_dict = checkpoint

        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)

        # load pretrained parameter, considering unmapped parameters
        init_parameter = autoencoder.state_dict()
        pretrained_parameter = {}
        for k, v in state_dict.items():
            if k in para_name_map_dict.keys():
                k = para_name_map_dict[k]
            pretrained_parameter[k] = v
        init_parameter.update(pretrained_parameter)
        autoencoder.load_state_dict(init_parameter)

        cls.check_model_para_change(autoencoder, "After Loading Pretrain Para,")

        autoencoder.init_model_parameter = copy.deepcopy(autoencoder.state_dict())

        if device is not None:
            autoencoder.to(device)
        return autoencoder

    def check_model_para_change(self, info=""):
        current_para = {k: v.to("cpu") for k, v in self.state_dict().items()}
        for i, j in self.init_model_parameter.items():
            j = j.to("cpu")
            if not t.equal(j, current_para[i]):
                logger.debug(f"{info} Changed Parameter: {i}")
            else:
                logger.debug(f"{info} Unchanged Parameter: {i}")


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function. This is useful for treating neurons as features.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x


class GatedAutoEncoder(AutoEncoder):
    """
    A one-layer autoencoder with a gated mechanism to learn binary masks for intermediate features.
    Inherits from AutoEncoder and adds a learnable gate applied to the encoded features.
    """

    def __init__(self, activation_dim, dict_size, tied=True, temperature=1):
        super().__init__(activation_dim, dict_size, tied)
        # Initialize gate parameters, zeros initially so sigmoid starts at 0.5
        self.gate = nn.Parameter(t.zeros(dict_size))
        self.temperature = temperature

        self.init_model_parameter = copy.deepcopy(self.state_dict())

    def encode(self, x):
        f = super().encode(x)

        # Compute gating values using Straight-Through Estimator (STE)
        gate_sigmoid = t.sigmoid(self.gate / self.temperature)
        # Binary mask during forward pass, but gradients use the continuous sigmoid
        # gate_mask = (gate_sigmoid >= 0.5).float()
        gate_mask = (gate_sigmoid > 0.5).float()
        # STE trick
        # When forward, mask using 0 and 1, since only using gate_mask
        # When backward, only using gradient from gate_sigmoid since gate_mask don't have gradient
        gate = gate_mask + gate_sigmoid - gate_sigmoid.detach()

        # Apply the gate to the features
        f = f * gate  # shape (batch, dict_size) multiplied element-wise

        return f


class MultipleGatedAutoEncoder(AutoEncoder):
    """
    A one-layer autoencoder with a gated mechanism to learn binary masks for intermediate features.
    Inherits from AutoEncoder and adds a learnable gate applied to the encoded features.
    """

    def __init__(self, activation_dim, dict_size, gate_num=2, tied=True, temperature=1):
        super().__init__(activation_dim, dict_size, tied)
        # Initialize gate parameters, zeros initially so sigmoid starts at 0.5
        self.gate_list = nn.ParameterList([nn.Parameter(t.zeros(dict_size)) for _ in range(gate_num)])
        self.temperature = temperature

        self.init_model_parameter = copy.deepcopy(self.state_dict())

    def encode(self, x):
        f = super().encode(x)

        f_list = []
        for gate in self.gate_list:
            # Compute gating values using Straight-Through Estimator (STE)
            gate_sigmoid = t.sigmoid(gate / self.temperature)
            # Binary mask during forward pass, but gradients use the continuous sigmoid
            # gate_mask = (gate_sigmoid >= 0.5).float()
            gate_mask = (gate_sigmoid > 0.5).float()
            # STE trick
            # When forward, mask using 0 and 1, since only using gate_mask
            # When backward, only using gradient from gate_sigmoid since gate_mask don't have gradient
            gate = gate_mask + gate_sigmoid - gate_sigmoid.detach()

            # Apply the gate to the features
            f_list.append(f * gate)  # shape (batch, dict_size) multiplied element-wise

        return f_list

    def decode(self, f_list):
        x_hat_list = [self.decoder(f) + self.bias for f in f_list]
        return x_hat_list
