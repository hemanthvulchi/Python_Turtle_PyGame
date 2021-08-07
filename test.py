import pongMath as pm
from math import cos, sin, pi
import torch
import torch.nn.functional as TF
import torch.nn as nn

sample_list = [2,3,5,7,9,4]
list_size = 6
x_points = [0] * 12
y_points = [0] * 12
pdistance = [0] * 12
angle = 0
i = 0
for i in range(0, 12):
    x_points[i] = i
    y_points[i] = i
    pdistance[i] = i
    print("iteration",i," angle:", angle, " x:", x_points[i], " y:", y_points[i])
    angle += 30

loss = nn.CrossEntropyLoss()

Y = torch.tensor([2])
 
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 7]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print("cross entropy -------------")
print(" Y      :", Y)
print(" y_good :", Y_pred_good)
print(" y_bad  :", Y_pred_bad)
print(" y size :", Y.size())
print(" y bad_s:",  Y_pred_good.size())

print(l1.item())
print(l2.item())
