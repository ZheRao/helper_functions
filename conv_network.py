import torch
from torch import nn
class three_layer_conv_network(nn.Module):
    def __init__(self,input_dimension:list,conv_dimensions:list,kernel_sizes:list,strides:list,pool_sizes:list,pool_strides:list,
                  dropout_prob:list, paddings:list, output_logit:int,dense_neuron:int):
        num_layer = 3
        # make sure the length of lists are 3 for 3 stacks of conv layers, and the datatype in the list are correct
        assert len(conv_dimensions) >= num_layer, "Not enough dimensions given for the network, must give " + str(num_layer)
        assert len(kernel_sizes) >= num_layer, "Not enough kernel sizes given for the network, must give " + str(num_layer)
        assert type(kernel_sizes[0]) == tuple, "kernel size not tuple"
        assert len(strides) >= num_layer, "Not enough strides given for the network, must give " + str(num_layer)
        assert len(pool_sizes) >= num_layer, "Not enough max pool sizes given for the network, must give " + str(num_layer)
        assert type(pool_sizes[0]) == tuple, "max pool size not tuple"
        assert len(pool_strides) >= num_layer, "Not enough max pool sizes given for the network, must give " + str(num_layer)
        assert type(pool_strides[0]) == tuple, "max pool strides not tuple"
        assert len(dropout_prob) >= num_layer+1, "Not enough dropout probabilities given for the network, must give " + str(num_layer+1) + " include dense"
        assert len(paddings) >=num_layer, "Not enough paddings given for the network, must give " + str(num_layer)
        assert len(input_dimension) == 4, "Input dimension is not 4"
        
        # initialize parameters
        self.input_dimension = input_dimension
        self.output_logit = output_logit
        
        # model architecture
        super().__init__()
        
        # convolutional layers
        self.conv_1 = nn.Sequential(
            nn.Conv2d(input_dimension[1],conv_dimensions[0],kernel_size=kernel_sizes[0],stride=strides[0],padding=paddings[0]),
            nn.BatchNorm2d(conv_dimensions[0]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_prob[0]),
            nn.MaxPool2d(kernel_size=pool_sizes[0],stride=pool_strides[0])
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(conv_dimensions[0],conv_dimensions[1],kernel_size=kernel_sizes[1],stride=strides[1],padding=paddings[1]),
            nn.BatchNorm2d(conv_dimensions[1]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_prob[1]),
            nn.Conv2d(conv_dimensions[1],conv_dimensions[1],kernel_size=kernel_sizes[1],stride=strides[1],padding=paddings[1]),
            nn.BatchNorm2d(conv_dimensions[1]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_prob[1]),
            nn.MaxPool2d(kernel_size=pool_sizes[1],stride=pool_strides[1])
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(conv_dimensions[1],conv_dimensions[2],kernel_size=kernel_sizes[2],stride=strides[2],padding=paddings[2]),
            nn.BatchNorm2d(conv_dimensions[2]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_prob[2]),
            nn.Conv2d(conv_dimensions[2],conv_dimensions[2],kernel_size=kernel_sizes[2],stride=strides[2],padding=paddings[2]),
            nn.BatchNorm2d(conv_dimensions[2]),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_prob[2]),
            nn.MaxPool2d(kernel_size=pool_sizes[2],stride=pool_strides[2])
        )
        self.flatten = nn.Flatten(start_dim = 1, end_dim = -1)
        self.size_after_flatten, x = self.sample_conv_forward() # dimension after flatten layer
        self.dense = nn.Sequential(
            nn.Linear(self.size_after_flatten,dense_neuron),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob[3]),
            nn.Linear(dense_neuron,self.output_logit)
        )
        self.sample_dense_forward(x)
        

    # random sample forward pass determine sizes of outputs through various layers
    def sample_conv_forward(self):
        print("This is a sample forward pass\n--------------------------------\n")
        a = self.input_dimension
        x = torch.randn((1,a[1],a[2],a[3]))
        print(f"input size: {x.shape}\n")
        # conv layers
        x = self.conv_1(x)
        print(f"after the first convolutional layer, output size: {x.shape}\n")
        x = self.conv_2(x)
        print(f"after the second convolutional layer, output size: {x.shape}\n")
        x = self.conv_3(x)
        b = x.shape
        print(f"after the third convolutional layer, output size: {b}\n")
        # flatten and dense layers
        c = b[1] * b[2] * b[3]
        x = self.flatten(x)
        print(f"after the flatten layer, output size: {x.shape}\n")
        return c,x
    
    def sample_dense_forward(self,x:torch.Tensor):
        print(f"Input for dense layer: {x.shape}\n")
        x = self.dense(x)
        print(f"after the dense layers, output size: {x.shape}, ready for loss calculation with logit transformation\n")
    
    # forward method
    def forward(self,x:torch.Tensor):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


class simple_two_layer_conv_network(nn.Module):
    def __init__(self,input_dimension:list,conv_dimensions:list,kernel_sizes:list,strides:list,pool_sizes:list,pool_strides:list,
                  dropout_prob:list, paddings:list, output_logit:int,dense_neuron:int):
        num_layer = 2
        # make sure the length of lists are 3 for 3 stacks of conv layers, and the datatype in the list are correct
        assert len(conv_dimensions) >= num_layer, "Not enough dimensions given for the network, must give " + str(num_layer)
        assert len(kernel_sizes) >= num_layer, "Not enough kernel sizes given for the network, must give " + str(num_layer)
        assert type(kernel_sizes[0]) == tuple, "kernel size not tuple"
        assert len(strides) >= num_layer, "Not enough strides given for the network, must give " + str(num_layer)
        assert len(pool_sizes) >= num_layer-1, "Not enough max pool sizes given for the network, must give " + str(num_layer-1)
        assert type(pool_sizes[0]) == tuple, "max pool size not tuple"
        assert len(pool_strides) >= num_layer-1, "Not enough max pool sizes given for the network, must give " + str(num_layer-1)
        assert type(pool_strides[0]) == tuple, "max pool strides not tuple"
        assert len(dropout_prob) >= num_layer, "Not enough dropout probabilities given for the network, must give " + str(num_layer) 
        assert len(paddings) >=num_layer, "Not enough paddings given for the network, must give " + str(num_layer)
        assert len(input_dimension) == 4, "Input dimension is not 4"
        
        # initialize parameters
        self.input_dimension = input_dimension
        self.output_logit = output_logit
        
        # model architecture
        super().__init__()
        
        # convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dimension[1],conv_dimensions[0],kernel_size=kernel_sizes[0],stride=strides[0],padding=paddings[0]),
            nn.ReLU(),
            nn.Conv2d(conv_dimensions[0],conv_dimensions[1],kernel_size=kernel_sizes[0],stride=strides[0],padding=paddings[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_sizes[0],stride=pool_strides[0]),
            nn.Dropout2d(p=dropout_prob[0])
        )
        self.flatten = nn.Flatten(start_dim = 1, end_dim = -1)
        self.size_after_flatten, x = self.sample_conv_forward() # dimension after flatten layer
        self.dense = nn.Sequential(
            nn.Linear(self.size_after_flatten,dense_neuron),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob[1]),
            nn.Linear(dense_neuron,self.output_logit)
        )
        self.sample_dense_forward(x)
        

    # random sample forward pass determine sizes of outputs through various layers
    def sample_conv_forward(self):
        print("This is a sample forward pass\n--------------------------------\n")
        a = self.input_dimension
        x = torch.randn((1,a[1],a[2],a[3]))
        print(f"input size: {x.shape}\n")
        # conv layers
        x = self.conv_layers(x)
        b = x.shape
        print(f"after the convolutional layers, output size: {b}\n")
        # flatten and dense layers
        c = b[1] * b[2] * b[3]
        x = self.flatten(x)
        print(f"after the flatten layer, output size: {x.shape}\n")
        return c,x
    
    def sample_dense_forward(self,x:torch.Tensor):
        print(f"Input for dense layer: {x.shape}\n")
        x = self.dense(x)
        print(f"after the dense layers, output size: {x.shape}, ready for loss calculation with logit transformation\n")
    
    # forward method
    def forward(self,x:torch.Tensor):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x