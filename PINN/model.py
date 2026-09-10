import torch
from torch import nn


#defining the architecture
class NN(nn.Module):
    def __init__(self,retrain_seed):
        super(NN,self).__init__()

        self.retrain_seed=retrain_seed
        self.l1=nn.Linear(2,128)
        self.l2=nn.Linear(128,128)
        self.l3=nn.Linear(128,64)
        self.l4=nn.Linear(64,32)
        self.l5=nn.Linear(32,1)
        self.activation=nn.Tanh()

    def forward(self,x):
        x=self.l1(x)
        x=self.activation(x)
        x=self.l2(x)
        x=self.activation(x)
        x=self.l3(x)
        x=self.activation(x)
        x=self.l4(x)
        x=self.activation(x)
        x=self.l5(x)
        return x
    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if type(m)==nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g=nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight,gain=g)
                nn.init.zeros_(m.bias)

        self.apply(init_weights)
