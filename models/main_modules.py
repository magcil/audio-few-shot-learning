import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch import nn
import torch
import torch.nn.functional as F

from utils.spectrogram_augmentations import SpecAugment


class EncoderModule(nn.Module):

    def __init__(self, encoder_str, model_config):
        super(EncoderModule, self).__init__()
        self.augmentation_module = SpecAugment()
        self.encoder_str = encoder_str
        self.encoder = get_backbone_model(encoder_name=self.encoder_str, model_config=model_config)

    def forward(self, spec):
        ## Get a fixed number of augmentations of x in x_list
        spec_list = self.augmentation_module.apply(spec)
        ## get_encoder
        encoded_features = []
        for x in spec_list:
            encoded_x = self.encoder(x)
            ## Encoded x will be of shape [batch_size,D]
            encoded_features.append(encoded_x)
        return encoded_features


def floor_power(num, divisor, power):
    """Performs what we call a floor power, a recursive fixed division process
        with a flooring between each time

    Args:
        num (int or float):The original number to divide from
        divisor (int or float): The actual divisor for the number
        power (int): How many times we apply this divide and then floor

    Returns:
        int: The numerical result of the floor division process
    """
    for _ in range(power):
        num = np.floor(num / divisor)
    return num


def conv_block(in_channels, out_channels, pool_dim):
    """Returns a convolutional block that performs a 3x3 convolution, ReLu
    activation and a variable max pooling

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        pool_dim (int or tuple): Pooling variable for both kernel and stride,
            mainly used to ensure models created can work with smaller/larger
            sequences without reducing dims too far.

    Returns:
        Torch nn module: The torch nn seuqntial object with conv, batchnorm, relu
            and maxpool
    """
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(),
                          nn.MaxPool2d(kernel_size=pool_dim, stride=pool_dim))
    return block


def conv_encoder(in_channels, hidden_channels, pool_dim):
    """Generates a convolutonal based encoder

    Args:
        in_channels (int): The inital number of input channels into the
            encoder
        hidden_channels (int): The number of hidden channels in the convolutional
            procedure
        pool_dim (int or tuple): Pooling stride and kernel variable

    Returns:
        Torch nn module: The torch encoding model
    """
    return nn.Sequential(
        conv_block(in_channels, hidden_channels, pool_dim),
        conv_block(hidden_channels, hidden_channels, pool_dim),
        conv_block(hidden_channels, hidden_channels, pool_dim),
        conv_block(hidden_channels, hidden_channels, pool_dim),
    )


class StandardCNN(nn.Module):

    def __init__(self, in_channels, trial_shape, hidden_channels, pool_dim, out_dim):
        """Standard CNN backbone for meta-learning applications

        Args:
            in_channels (int): Number of input channels for the data
            trial_shape (tuple or array)): An example data sample shape array/tuple,
                used to work out the input to the final linear layer
            hidden_channels (int): Number of hidden channels used thoughout the
                 main encoder structure
            pool_dim (int or tuple): Pooling stride and kernel variable
            out_dim (int): Number of nodes to output to in final linear layer
        """
        super(StandardCNN, self).__init__()
        self.conv_encoder = conv_encoder(in_channels, hidden_channels, pool_dim)

        # Caluclates how many nodes needed to collapse from conv layer

        num_logits = int(64 * floor_power(trial_shape[2], pool_dim[0], 4) * floor_power(trial_shape[3], pool_dim[1], 4))
        self.logits = nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(num_logits, eps=1e-05, momentum=0.1, affine=True),
                                    nn.Linear(in_features=num_logits, out_features=out_dim))

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Trainable Params: {self.params}')

    def forward(self, x):
        x = self.conv_encoder(x)
        x = x.view(x.size(0), -1)
        return self.logits(x)


class StandardHybrid(nn.Module):

    def __init__(self, in_channels, seq_layers, seq_type, bidirectional, hidden_channels, pool_dim, out_dim):
        """Standardised conv-seq hybrid base learner. Shares a base convolutional
            encoder with the standrdised CNN

        Args:
            in_channels (int): Number of input channels for the data
            seq_layers (int): Number of layers to use in teh sequential part
            seq_type (str): The sequential layer type to use
            bidirectional (boolean): Whether the seq model part should be bidirectional
            hidden_channels (int): Number of hidden channels in teh conv encoder
            pool_dim (int or tuple): MaxPool kernel and stride
            out_dim (int): Number of logits to output to

        Raises:
            ValueError: Error raised if sequential layer type not in ['LSTM',
                'GRU', 'RNN']
        """
        super(StandardHybrid, self).__init__()

        self.bidirectional = bidirectional
        self.seq_type = seq_type

        # This is the number of output channels * floor_div(n_mels, pool, 4)
        hidden = 64
        # Convolutional base encoder
        self.conv_encoder = conv_encoder(in_channels, hidden_channels, pool_dim)

        # Make sure value enetered is reasonable
        if seq_type not in ['LSTM', 'GRU', 'RNN']:
            raise ValueError('Seq type not recognised')

        # Generates the sequential layer call
        seq_layer_call = getattr(nn, seq_type)
        self.seq_layers = seq_layer_call(input_size=hidden,
                                         hidden_size=hidden,
                                         num_layers=seq_layers,
                                         bidirectional=bidirectional,
                                         batch_first=True)

        # We enforce having a final linear layer with batch norm for ocnvergence
        self.logits = nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(hidden, eps=1e-05, momentum=0.1, affine=True),
                                    nn.Linear(in_features=hidden, out_features=out_dim))

        # Count and print the number of trainable parameters
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'Num Layers: {seq_layers} -> Trainable Params: {self.params}')

    def many_to_one(self, t, lengths):
        return t[torch.arange(t.size(0)), lengths - 1]

    def forward(self, x):
        x = self.conv_encoder(x)
        # (batch, time, freq, channel)
        x = x.transpose(1, -1)
        batch, time = x.size()[:2]

        #(batch, time, channel*freq)
        x = x.reshape(batch, time, -1)

        # Pass through the sequential layers
        if self.seq_type == 'LSTM':
            output, (hn, cn) = self.seq_layers(x)
        else:
            output, hn = self.seq_layers(x)

        forward_output = output[:, :, :self.seq_layers.hidden_size]
        backward_output = output[:, :, self.seq_layers.hidden_size:]

        # g(x_i, S) = h_forward_i + h_backward_i + g'(x_i)
        # AKA A skip connection between inputs and outputs is used
        if self.bidirectional:
            x = forward_output + backward_output + x
        else:
            x = forward_output + x

        x = self.many_to_one(x, x.shape[-2])
        x = self.logits(x)

        return x


class SelfAttention(nn.Module):

    def __init__(self, model_config):
        super(SelfAttention, self).__init__()
        self.embed_dim = model_config['Attention']['embed_dim']
        self.num_heads = model_config['Attention']['num_heads']
        self.ffn_dim = model_config['Attention']['ffn_dim']
        self.dropout = model_config['Attention']['dropout']

        # TransformerEncoderLayer: includes MultiheadAttention, FeedForward, and LayerNorm
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            batch_first=True  # Makes the layer expect input as (batch_size, seq_length, embed_dim)
        )

    def forward(self, x):
        # Input x shape: (batch_size, 4, D)

        # Pass input through TransformerEncoderLayer
        attn_output = self.encoder_layer(x)  # Output shape: (batch_size, 4, D)

        # Channel-wise concatenation: Concatenate along the feature dimension to get (batch_size, 4 * D)
        output = torch.cat([attn_output[:, i, :] for i in range(attn_output.size(1))], dim=-1)

        return output


class ProjectionHead(nn.Module):

    def __init__(self, model_config):
        super(ProjectionHead, self).__init__()
        self.input_dim = model_config['Projection']['input_dim']
        self.hidden_dim = model_config['Projection']['hidden_dim']
        self.output_dim = model_config['Projection']['output_dim']

        # Single layer: Keep the same dimension for input and output
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)

        # Use Layer Normalization instead of Batch Normalization
        self.ln1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.ln2 = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        # Pass through the first layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        # Pass through the second layer
        x = self.fc2(x)
        x_norm = F.normalize(x, p=2.0, dim=1, eps=1e-12, out=None)

        return x_norm


def get_backbone_model(encoder_name, model_config):

    if encoder_name == 'CNN':
        in_channels = model_config[encoder_name]['in_channels']
        hidden_channels = model_config[encoder_name]['hidden_channels']
        pool_dim = model_config[encoder_name]["pool_dim"]
        out_dim = model_config[encoder_name]["out_dim"]
        backbone_model = StandardCNN(in_channels=in_channels,
                                     hidden_channels=hidden_channels,
                                     pool_dim=pool_dim,
                                     out_dim=out_dim)

    elif encoder_name == 'Hybrid':
        in_channels = model_config[encoder_name]["in_channels"]
        seq_layers = model_config[encoder_name]["seq_layers"]
        seq_type = model_config[encoder_name]["seq_type"]
        bidirectional = model_config[encoder_name]["bidirectional"]
        hidden_channels = model_config[encoder_name]['hidden_channels']
        pool_dim = model_config[encoder_name]["pool_dim"]
        out_dim = model_config[encoder_name]['out_dim']
        backbone_model = StandardHybrid(in_channels=in_channels,
                                        seq_layers=seq_layers,
                                        seq_type=seq_type,
                                        bidirectional=bidirectional,
                                        hidden_channels=hidden_channels,
                                        pool_dim=pool_dim,
                                        out_dim=out_dim)
    return backbone_model
