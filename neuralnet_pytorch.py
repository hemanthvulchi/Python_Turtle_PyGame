import torch


# pytorch two layer network
class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h_relu1 = self.linear1(x).clamp(min=0)
        h_relu2 = self.linear2(h_relu1).clamp(min=0)
        y_pred = self.linear3(h_relu2).sigmoid()
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
        self.criterion = torch.nn.MSELoss(reduction='sum').cuda("cuda:0")
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-10)

    def get_input(self, x1, x2, x3, x4):
        self.x.data = torch.tensor([x1, x2, x3, x4], device=self.device, dtype=self.ndtype).data

    def get_output(self, y1, y2, y3, y4):
        self.y = torch.tensor([y1, y2, y3, y4], device=self.device, dtype=self.ndtype).data

    # observe the world - get distances from the agent
    def observe_world(self, a_distances):
        #self.x = a_distances
        self.get_input(a_distances[0], a_distances[1], a_distances[2], a_distances[3])
        

    def action_world(self):
        self.y_pred = self.model(self.x).cuda("cuda:0")
        return self.maximum_index()

    def observe_result(self):
        self.y.zero_()
        y_index = self.minimum_index()
        if y_index == 0:
            self.y.data = torch.tensor([1, 0, 0, 0], device=self.device, dtype=self.ndtype).data
        elif y_index == 1:
            self.y.data = torch.tensor([0, 1, 0, 0], device=self.device, dtype=self.ndtype).data
        elif y_index == 2:
            self.y.data = torch.tensor([0, 0, 1, 0], device=self.device, dtype=self.ndtype).data
        elif y_index == 3:
            self.y.data = torch.tensor([0, 0, 0, 1], device=self.device, dtype=self.ndtype).data
        self.loss = self.criterion(self.y_pred, self.y).cuda("cuda:0")
        print(" y pred  :", self.y_pred)
        print(" y actual:", self.y)
        self.optimizer.zero_grad
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def minimum_index(self):
        minimum_value = 10000
        for i in range(self.D_in):
            if minimum_value > self.x[i]:
                minimum_value = self.x[i]
                minimum_index = i
        return minimum_index

    def maximum_index(self):
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


# gameengine = NeuralNetCustom()
# gameengine.train()
