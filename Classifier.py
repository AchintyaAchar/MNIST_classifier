import torch.nn as nn
import torch 
print('classifier module successfully imported!')

class Conv_network(nn.Module): 
    def __init__(self, in_channels: int, out_prediction_size: int, network_layer_shape: tuple[int, int, int, int]):

        super().__init__()

        self.in_channels = in_channels
        self.internal_layers = []

        # defining the first layer with the in_channels
        self.internal_layers.append(nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, padding=2))
        self.internal_layers.append(nn.BatchNorm2d(self.internal_layers[-1].out_channels))
        self.internal_layers.append(nn.ReLU())

        # iterates over the given repeats for each layer size
        for j, layer_repeats in enumerate(network_layer_shape):

            # iterates over each individual repeat 
            for i in range(layer_repeats):
                if i == 0:
                    padding_adjust = 0
                else:
                    padding_adjust = (self.internal_layers[-3].kernel_size[0]-1)//2

                self.internal_layers.append(nn.Conv2d(in_channels=self.internal_layers[-3].out_channels, out_channels = (j+1)*8, kernel_size=3, padding=padding_adjust))
                self.internal_layers.append(nn.BatchNorm2d(self.internal_layers[-1].out_channels))
                self.internal_layers.append(nn.ReLU())
        
        self.internal_layers = nn.ModuleList(self.internal_layers)
        self._valid_cnn()

        self.gap = nn.AdaptiveAvgPool2d((2,2))
        

        self.end_perceptron = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                self.internal_layers[-3].out_channels
                * self.gap.output_size[0]
                * self.gap.output_size[1],
                30,
            ),
            nn.ReLU(),
            nn.Linear(30, out_prediction_size)
        )

        
        

        


    def _conv_only_forward(self, x):

        for layer in self.internal_layers:
            x = layer(x)

        return x


    def _valid_cnn(self):
        """Returns the output dimentions of the defined cnn and returns none if
            not valid."""
        
        mock_tensors = torch.randn(1,self.in_channels, 28,28)

        try:
            outp = self._conv_only_forward(mock_tensors)
        except RuntimeError as re:
            print(f'convolutional network is not valid, there was a runtime error')
            raise

        except Exception as err:
            print(f'there was a {type(err).__name__} error')
            raise

    def forward(self, x):
        conv_out = self._conv_only_forward(x)
        perceptron_in = self.gap(conv_out)

        return self.end_perceptron(perceptron_in)


if __name__ == "__main__":

    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('using apple\'s gpu!')
    else:
        device = torch.device('cpu')
        print('unable to accelerate, using cpu')

    

    test_net = Conv_network(1, 10, (3,1,1,1) ).to(device)

    test_input = torch.randn((2,1,28,28)).to(device)

    output = test_net(test_input)

    print(f'model outputted a tensor with shape: {output.shape}')
    

    

        








    



        

