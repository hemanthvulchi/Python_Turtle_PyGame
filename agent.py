import turtle
import random


class Agent():
    def __init__(self):
        self.agent=turtle.Turtle()
        self.agent.speed(0)
        self.agent.shape("circle")
        self.agent.color("black")
        self.agent.penup()
        self.agent.goto(random.randrange(-390,390),random.randrange(-390,390))
        self.agent.speed(2)
        self.agent.dx =0.05
        
    def move(self):
        self.agent.setx(self.agent.xcor()+self.agent.dx)


