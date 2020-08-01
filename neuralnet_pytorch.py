import torch
import torch.nn.functional as TF
import matplotlib.pyplot as plt
import numpy as np


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
        #y_pred = self.linear3(x).sigmoid()
        y_pred = self.linear3(x).tanh()
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
        self.D_in = 4
        self.H1 = 4
        self.H2 = 4
        self.D_out = 4

        # Setting input and output tensors
        self.x = torch.randn(self.N, self.D_in, device=self.device, dtype=self.ndtype)
        self.y = torch.randn(self.N, self.D_out, device=self.device, dtype=self.ndtype)

        # Initializing models, criterion and optimizer
        self.model = ThreeLayerNet(self.D_in, self.H1, self.H2, self.D_out)
        self.model.cuda("cuda:0")
        self.weights_init(self.model)
        #self.model.weight.data.uniform_(0.0, 1.0)
        self.criterion = torch.nn.MSELoss(reduction='sum').cuda("cuda:0")
        #self.criterion = torch.nn.NLLLoss(reduction='sum').cuda("cuda:0")
        #self.criterion = torch.nn.SmoothL1Loss().cuda("cuda:0")
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform(m.weight.data)
            torch.nn.init.xavier_uniform(m.bias.data)

    def get_input(self, x1, x2, x3, x4):
        self.np_inputs = np.array([x1, x2, x3, x4])
        self.norm = np.linalg.norm(self.np_inputs)
        np_inputs = self.np_inputs / self.norm
        lowest_distance = min(np_inputs)
        np_inputs = np_inputs - lowest_distance

        self.x.data = torch.tensor(np_inputs, device=self.device, dtype=self.ndtype).data
        # self.x.data = torch.tensor([x1, x2, x3, x4], device=self.device, dtype=self.ndtype).data

    def get_output(self, y1, y2, y3, y4):
        self.y = torch.tensor([y1, y2, y3, y4], device=self.device, dtype=self.ndtype).data

    # observe the world - get distances from the agent
    def observe_world(self, a_distances):
        # self.x = a_distances
        self.get_input(a_distances[0], a_distances[1], a_distances[2], a_distances[3])
        
    # neural network forward function; to be used after observe world function || combine with observe_world later    
    def action_world(self, a_distances):
        # initialize distances
        self.get_input(a_distances[0], a_distances[1], a_distances[2], a_distances[3])
        self.y_pred = self.model(self.x).cuda("cuda:0")
        return self.highest_probability_index()

    # observe result is to calculate the losss based on the movement and calculates the loss function
    def observe_result(self, iteration):
        self.y.zero_()
        y_index = self.minimum_index()
        # if y_index == 0:
        #     self.y.data = torch.tensor([1, 0, 0, 0], device=self.device, dtype=self.ndtype).data
        # elif y_index == 1:
        #     self.y.data = torch.tensor([0, 1, 0, 0], device=self.device, dtype=self.ndtype).data
        # elif y_index == 2:
        #     self.y.data = torch.tensor([0, 0, 1, 0], device=self.device, dtype=self.ndtype).data
        # elif y_index == 3:
        #     self.y.data = torch.tensor([0, 0, 0, 1], device=self.device, dtype=self.ndtype).data
        if y_index == 0:
            self.y.data = torch.tensor([1, -1, -1, -1], device=self.device, dtype=self.ndtype).data
        elif y_index == 1:
            self.y.data = torch.tensor([-1, 1, -1, -1], device=self.device, dtype=self.ndtype).data
        elif y_index == 2:
            self.y.data = torch.tensor([-1, -1, 1, -1], device=self.device, dtype=self.ndtype).data
        elif y_index == 3:
            self.y.data = torch.tensor([-1, -1, -1, 1], device=self.device, dtype=self.ndtype).data

        self.loss = self.criterion(self.y_pred, self.y).cuda("cuda:0")
        self.optimizer.zero_grad
        self.loss.backward()
        if iteration % 100 == 99:
            print("================================================================================================")
            print("iteration:", iteration)
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
        self.optimizer.step()
        if iteration % 100 == 99:
            print(" x input :", self.x)
            print(" y actual:", self.y)
            print(" y pred  :", self.y_pred)
            print(" loss    :", self.loss.item())
        return self.loss.item()

    # is used to return the minimum index of a array || used for finding the minimum distance direction of the agent
    def minimum_index(self):
        minimum_value = 10000
        for i in range(self.D_in):
            if minimum_value > self.x[i]:
                minimum_value = self.x[i]
                minimum_index = i
        return minimum_index

    # is used to return the highest probablity index
    def highest_probability_index(self):
        maximum_value = 0
        for i in range(self.D_in):
            if maximum_value < self.y_pred[i]:
                maximum_value = self.y_pred[i]
                maximum_index = i
        return maximum_index

    def train(self):
        for t in range(10000):
            self.y_pred = self.model(self.x).cuda("cuda:0")
            # model.cuda("cuda:0")
            self.loss = self.criterion(self.y_pred, self.y.sigmoid()).cuda("cuda:0")
            if t % 100 > 97:
                print(t, self.loss.item())
            self.optimizer.zero_grad
            self.loss.backward()
            self.optimizer.step()

    def plot_grad_flow(self, named_parameters):
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(ave_grads))
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.show()


# gameengine = NeuralNetCustom()
# gameengine.train()

