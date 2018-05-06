import torch
import torch.functional as F
from torch import nn


class BaseLine (nn.Module):
    def __init__(self, image_channels=1):
        super(BaseLine, self).__init__()

        self.conv1 = nn.Conv2d(image_channels, 256, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=5, stride=1)
        self.dense1 = nn.Linear(328)
        self.dense2 = nn.Linear(192)
        self.dense3 = nn.Linear(10)

    def forward(self, images):
        out = F.relu(self.conv1(images), inplace=False)
        out = F.relu(self.conv2(out), inplace=False)
        out = F.relu(self.conv3(out), inplace=False)
        out = torch.view(out.shape[0], -1)
        out = F.relu(self.dense1(out), inplace=False)
        out = F.dropout(F.relu(self.dense2(out), inplace=False), p=0.5)
        out = F.softmax(self.dense3(out), dim=-1)

        return out


class CapsuleLayer(nn.Module):
    """
    TODO add very long doc
    """
    def __init__(self, input_units, input_channels, num_units, channels_per_unit,
                 kernel_size, stride, routing, routing_iterations):

        super(CapsuleLayer, self).__init__()
        self.input_units = input_units
        self.input_channels = input_channels
        self.num_units = num_units
        self.channels_per_unit = channels_per_unit
        self.kernel_size = kernel_size
        self.stride = stride
        self.routing = routing
        self.routing_iterations = routing_iterations

        if self.routing:
            """
            'W_ij is a weight matrix between each u_i, for i in (1, 32x6x6) in PrimaryCapsules and v_j,
             for j in (1, 10)'
            Additionally, W_ij is an (8, 16) matrix.
            This means the layer will have a parameter matrix of size (input_units * H_in * W_in, num_classes, 
            input_channels, channels_per_unit). To make it easier for us to define this matrix, let us assumme 
            `input_units == original_input_units * H_in * W_in` when routing is active.
            """
            self.weights = nn.Parameter(torch.randn(input_units, num_units, input_channels, channels_per_unit))
        else:
            """
            For the PrimaryCaps layer (if the previous layer is not capsular too), the output should be the same as 
            using multiple small convolutional layers. Using a ModuleList facilitates interaction with all the units in 
            a pythonic way. Section 4,  3rd paragraph, describes the PrimaryCaps layer as having 32 units, each with 8 
            channels, with 9x9 kernel and stride 2.
            """
            self.units = nn.ModuleList([nn.Conv2d(input_channels, channels_per_unit, kernel_size, stride)
                                        for _ in range(self.num_units)])

    def forward(self, inputs):
        """
        Decide between applying routing or plain convolutions.
        Routing is only used if between 2 consecutive layers
        TODO try to implement routing as a method of the network and not the layers
        """
        if self.routing:
            return self._routing(inputs)
        else:
            return self._apply_conv_units(inputs)

    def _routing(self, inputs):
        """
        TODO add doc
        This function is probably rather heavy. Should try profiling.
        """
        batch_size = inputs.data.shape[0]
        weights = torch.stack([self.weights] * batch_size, dim=0)

        current_votes = inputs.permute([0, 2, 1])
        current_votes = torch.stack([current_votes] * self.num_units, dim=2)
        current_votes = torch.stack([current_votes] * self.channels_per_unit, dim=-1)

        logits = Variable(torch.zeros(current_votes.data.shape))
        pondered_votes = weights * current_votes  # Uji

        for iteration in range(self.routing_iterations):
            couplings = F.softmax(logits, dim=-1)
            out = couplings * pondered_votes
            out = squash(out)
            agreement = pondered_votes * out
            logits = logits + agreement

        out = out.permute([0, 2, 1, 3, 4])
        return out

    def _apply_conv_units(self, inputs):
        """
        Shape: (batch_size, input_channels, H, W) -> (batch_size, units, channels_per_unit, H', W')
        H' and W' can be calculated using standard formulae for convolutional outputs
        """
        output = [unit(inputs) for unit in self.units]
        output = torch.stack(output, dim=1)  # New dimension 1 will have size `units`
        return output


class CapsNet(nn.Module):
    def __init__(self, conv_in_channels=1, conv_out_channels=256, conv_kernel_size=9, conv_stride=1,
                 primary_units=32, primary_dim=8, primary_kernel_size=9, primary_stride=2,
                 num_classes=10, digits_dim=16, dense_units_1=512, dense_units_2=1024, dense_units_3=784,
                 routing_iterations=1):
        """
        TODO Add very long doc for this...
        dense_units_3 : int, number of pixels in an input image
        """
        super(CapsNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=conv_in_channels,
                               out_channels=conv_out_channels,
                               kernel_size=conv_kernel_size,
                               stride=conv_stride)
        self.primary_caps = CapsuleLayer(input_units=None,
                                         input_channels=conv_out_channels,
                                         num_units=primary_units,
                                         channels_per_unit=primary_dim,
                                         kernel_size=primary_kernel_size,
                                         stride=primary_stride,
                                         routing=False,
                                         routing_iterations=routing_iterations)
        self.digits_caps = CapsuleLayer(input_units=6*6*primary_units,
                                        input_channels=primary_dim,
                                        num_units=num_classes,
                                        channels_per_unit=digits_dim,
                                        kernel_size=0,
                                        stride=0,
                                        routing=True,
                                        routing_iterations=routing_iterations)
        self.decoder = nn.Sequential(OrderedDict([
                                                  ('decoder1', nn.Linear(num_classes * digits_dim, dense_units_1)),
                                                  ('relu1', nn.ReLU()),
                                                  ('decoder2', nn.Linear(dense_units_1, dense_units_2)),
                                                  ('relu2', nn.ReLU()),
                                                  ('decoder3', nn.Linear(dense_units_2, dense_units_3)),
                                                  ('decoder_out', nn.Sigmoid())
                                                ]))

    def forward(self, images):
        """
        Receives batch of images and outputs log probabilities of each class for each image in the batch.

        Parameters:
            images : Variable containing Tensor of shape (batch_size, num_channels, H, W)

        Returns:
            out : Variable containing Tensor of shape(batch_size, num_classes
        """
        batch_size = images.shape[0]

        conv_out = self.conv0(images)
        conv_out = F.relu(conv_out, inplace=False)

        primary_caps_out = self.primary_caps(conv_out)
        squashed_primary_out = squash(primary_caps_out)

        # -> (batch_size, primary_units, )
        digit_in = squashed_primary_out.view(batch_size, self.primary_caps.channels_per_unit, -1)
        digit_out = self.digits_caps(digit_in)

        out = digit_out
        while len(out.shape) > 2:
            out = torch.norm(out, dim=-1)

        return out
