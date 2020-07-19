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


class NeuralNetPytorch():
    # initialize - helps in creating the tensor
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")          
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

        dtype = torch.float

        self.N = 1
        self.D_in = 4
        self.H1 = 4
        self.H2 = 4
        self.D_out = 4

        self.x = torch.randn(self.N, self.D_in, device=self.device, dtype=dtype)
        self.y = torch.randn(self.N, self.D_out, device=self.device, dtype=dtype)

    def get_input(self, x1, x2, x3, x4):
        self.x = torch.tensor([x1, x2, x3, x4])

    def get_output(self, y1, y2, y3, y4):
        self.y = torch.tensor([y1, y2, y3, y4])

    # observe
    def observe_world(self, a_distances):
        self.agent_distances = a_distances
        
    def train(self):
        self.model = ThreeLayerNet(self.D_in, self.H1, self.H2, self.D_out)
        self.model.cuda("cuda:0")
        self.criterion = torch.nn.MSELoss(reduction='sum').cuda("cuda:0")
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

        for t in range(10000):
            self.y_pred = self.model(self.x).cuda("cuda:0")
            # model.cuda("cuda:0")
            self.loss = self.criterion(self.y_pred, self.y.sigmoid()).cuda("cuda:0")
            if t % 100 > 97:
                print(t, self.loss.item())
            self.optimizer.zero_grad
            self.loss.backward()
            self.optimizer.step()


# gameengine = NeuralNetPytorch()
# gameengine.train()
