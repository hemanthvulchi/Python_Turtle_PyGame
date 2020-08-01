#this contains powersource information
import turtle
import random
import math

class PowerSource():
    def __init__(self,screen_world):
        self.psource = turtle.Turtle()
        self.psource.speed(0)
        self.psource.shape("circle")
        self.psource.color("yellow")
        self.psource.penup()
        self.psource.goto(random.randrange(-(screen_world.size_x),screen_world.size_x),
                          random.randrange(-(screen_world.size_x),screen_world.size_x))
        self.psource.goto(20,0)
        self.psource.shapesize(stretch_len=0.1,stretch_wid=0.1)
        self.psource.speed(2)
        self.psource.dx = 0.05
        #self.size_x = screen_world.size_x
        #self.size_y = screen_world.size_y
        
    #move powersource if the agent touches the power source
    def check_agent(self,turtle_agent):
        self.x_power = self.psource.xcor()
        self.y_power = self.psource.ycor()
        self.x_agent = turtle_agent.agent.xcor()
        self.y_agent = turtle_agent.agent.ycor()
        if self.point_distance(self.x_power,self.y_power,self.x_agent,self.y_agent) < 21:
            self.psource.goto(random.randrange(-390,390),random.randrange(-390,390))
            print("Powersource is regenerated")

    #function to find the distance between two points
    def point_distance(self,x1,y1,x2,y2):  
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return dist  

    def goto_random(self):
        self.psource.goto(random.randrange(-390,390),random.randrange(-390,390))

    def move(self):
        self.psource.setx(self.psource.xcor()+self.psource.dx)


