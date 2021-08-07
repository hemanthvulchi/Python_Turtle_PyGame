import torch
import torch.nn.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import pongMath as pm

# pytorch two layer network
class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        #self.linear1 = torch.nn.Linear(D_in, H1)
        #self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        #h_relu1 = self.linear1(x).clamp(min=0)
        #h_relu2 = self.linear2(h_relu1).clamp(min=0)
        # y_pred = self.linear3(h_relu2)
        #return y_pred
        #h_relu1 = TF.relu(self.linear1(x))
        #h_relu2 = TF.relu(self.linear2(h_relu1))
        #y_pred = self.linear3(h_relu1)
        y_pred = self.linear3(x)
        #y_pred = self.linear3(x).softmax(dim=1)
        #y_pred = self.linear3(x).log_softmax
        #y_pred = TF.relu(self.linear3(x))
        #y_pred = self.linear3(h_relu1).tanh()
        return y_pred


class NeuralNetCustom():
    # initialize - helps in creating the tensor
    def __init__(self):

        # Checking if GPU is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        # Setting number of batches, inputs, hidden layers and outputs
        self.ndtype = torch.float
        self.N = 1
        self.D_in = 12
        self.H1 = 4
        self.H2 = 4
        self.D_out = 4

        # Setting input and output tensors
        self.x = torch.randn(self.N, self.D_in, device=self.device, dtype=self.ndtype)
        self.y = torch.randn(self.N, self.D_out, device=self.device, dtype=self.ndtype)

        # Initializing models, criterion and optimizer
        self.model = ThreeLayerNet(self.D_in, self.H1, self.H2, self.D_out)
        #self.model.cuda("cuda:0")
        self.model.to(self.device)
        self.weights_init(self.model)
        #self.model.weight.data.uniform_(0.0, 1.0)
        #self.criterion = torch.nn.MSELoss(reduction='sum').cuda("cuda:0")
        self.criterion = torch.nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    # initialize the weights 
    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)
            torch.nn.init.xavier_uniform(m.bias.data)

    # use the inputs entered and normalize them and enter them in a tensor
    def get_input(self, x_inputs):
        self.np_inputs = pm.PongMath.normalize(x_inputs)
        self.x.data = torch.tensor(self.np_inputs, device=self.device, dtype=self.ndtype).data

    def get_output(self, y1, y2, y3, y4):
        self.y = torch.tensor([[y1], [y2], [y3], [y4]], device=self.device, dtype=self.ndtype).data

    # neural network forward function; to be used after observe world function || combine with observe_world later    
    def action_world(self, a_distances):
        self.get_input(a_distances)
        self.y_pred = self.model(self.x)
        #print("print in action world:", self.y_pred)
        #print("print in action world:", self.y_pred.size)

        return pm.PongMath.maximum_index(self.y_pred, self.D_out)

    # custom loss function
    def my_loss(self, output, target):
        loss = torch.mean((output - target)**2)
        return loss

    # observe result is to calculate the losss based on the movement and calculates the loss function
    def observe_result(self, iteration, y_index):
        self.y.zero_()
        # if y_index == 0:
        #     self.y.data = torch.tensor([1, 0, 0, 0], device=self.device, dtype=self.ndtype).data
        # elif y_index == 1:
        #     self.y.data = torch.tensor([0, 1, 0, 0], device=self.device, dtype=self.ndtype).data
        # elif y_index == 2:
        #     self.y.data = torch.tensor([0, 0, 1, 0], device=self.device, dtype=self.ndtype).data
        # elif y_index == 3:
        #     self.y.data = torch.tensor([0, 0, 0, 1], device=self.device, dtype=self.ndtype).data            
        if y_index == 0:
            self.y.data = torch.tensor([0], device=self.device).data
        elif y_index == 1:
            self.y.data = torch.tensor([1], device=self.device).data
        elif y_index == 2:
            self.y.data = torch.tensor([2], device=self.device).data
        elif y_index == 3:
            self.y.data = torch.tensor([3], device=self.device).data            
        self.y_pred_reshape = torch.reshape(self.y_pred, (1, 4))
        self.y_pred_reshape.to(device = self.device)
        # print(" yrs size:", self.y_pred_reshape.size())
        # print(" yrs     :", self.y_pred_reshape)        
        #self.y_label = torch.argmax(self.y, 1)
        #print(" yl size  :", self.y_label.size())
        #print(" yl pred_s:", self.y_label.size())
        #self.loss = self.criterion(self.y_pred, self.y_label)
        self.loss = self.criterion(self.y_pred_reshape, self.y)
        #self.loss = self.my_loss(self.y_pred, self.y)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        if iteration % 100 == 99:
            print("================================================================================================")
            print("iteration:", iteration)
            print("Class:",self.model)
            #for name, param in self.model.named_parameters():
            #    print(name, param, param.grad)
            #self.model.linear1.register_forward_hook(lambda grad: print(grad))
            #self.model.linear1.register_backward_hook(lambda grad: print(grad))
            #print("Linear 1 Wg:", self.model.linear1.weight)
            #print("Linear 1 Gd:", self.model.linear1.weight.grad)
            #print("Linear 1 Bi:", self.model.linear1.bias)
            #print("Linear 1 Gd:", self.model.linear1.bias.grad)
            #print("Linear 2 Wg:", self.model.linear2.weight)
            #print("Linear 2 Gd:", self.model.linear2.weight.grad)
            #print("Linear 2 Bi:", self.model.linear2.bias)
            #print("Linear 2 Gd:", self.model.linear2.bias.grad)
            print("Linear 3 Wg:", self.model.linear3.weight)
            print("Linear 3 Gd:", self.model.linear3.weight.grad)
            print("Linear 3 Bi:", self.model.linear3.bias)
            print("Linear 3 Gd:", self.model.linear3.bias.grad)
            #self.plot_grad_flow(self.model.named_parameters())

        if iteration % 100 == 99:
            print(" x input :", self.x)
            print(" y actual:", self.y)
            print(" y pred  :", self.y_pred)
            print(" loss    :", self.loss.item())
            print("y size     :", self.y.size())
            print("y pred size:", self.y_pred.size())
        return self.loss.item()

    # is used to return the minimum index of a array || used for finding the minimum distance direction of the agent
 
    # def plot_grad_flow(self, named_parameters):
    #     ave_grads = []
    #     layers = []
    #     for n, p in named_parameters:
    #         if(p.requires_grad) and ("bias" not in n):
    #             layers.append(n)
    #             ave_grads.append(p.grad.abs().mean())
    #     plt.plot(ave_grads, alpha=0.3, color="b")
    #     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(xmin=0, xmax=len(ave_grads))
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)
    #     plt.show()


# gameengine = NeuralNetCustom()
# gameengine.train()

