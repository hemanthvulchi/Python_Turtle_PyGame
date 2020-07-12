import turtle
import random


class PowerSource():
    def __init__(self):
        self.psource=turtle.Turtle()
        self.psource.speed(0)
        self.psource.shape("circle")
        self.psource.color("yellow")
        self.psource.penup()
        self.psource.goto(random.randrange(-390,390),random.randrange(-390,390))
        self.psource.speed(2)
        self.psource.dx =0.05
        
    def move(self):
        self.psource.setx(self.psource.xcor()+self.psource.dx)


