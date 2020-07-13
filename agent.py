#this contains the agent itself and the functions it can perform
import turtle
import random
import math

class Agent():
    radius=20

    #
    def __init__(self):
        self.agent=turtle.Turtle()
        self.agent.speed(0)
        self.agent.shape("circle")
        self.agent.shapesize(stretch_wid=2,stretch_len=2)
        self.agent.color("black")
        self.agent.penup()
        #self.agent.goto(random.randrange(-390,390),random.randrange(-390,390))
        self.agent.goto(0,0)
        self.agent.speed(2)
        self.agent.dx =0.5
        self.agent.dy =0.5
        

    def calculate_distance(self,turtle_power):
        self.x_top = self.agent.xcor()
        self.y_top = self.agent.ycor() + self.radius 
        self.x_bot = self.agent.xcor()
        self.y_bot = self.agent.ycor() - self.radius 
        self.x_right = self.agent.xcor() + self.radius
        self.y_right = self.agent.ycor()  
        self.x_left = self.agent.xcor() - self.radius
        self.y_left = self.agent.ycor()  
        self.x_power = turtle_power.psource.xcor()
        self.y_power = turtle_power.psource.ycor()
        self.distance_top= self.point_distance(self.x_top,self.y_top,self.x_power,self.y_power)
        self.distance_bot= self.point_distance(self.x_bot,self.y_bot,self.x_power,self.y_power)
        self.distance_right= self.point_distance(self.x_right,self.y_right,self.x_power,self.y_power)
        self.distance_left= self.point_distance(self.x_left,self.y_left,self.x_power,self.y_power)
        self.distance_center= self.point_distance(self.agent.xcor(),self.agent.ycor(),self.x_power,self.y_power)        

    def print_distance(self):
        print ("================")
        print ("x cor of agent is ", self.agent.xcor())
        print ("y cor of agent is ", self.agent.ycor())
        print ("x cor of power is ", self.x_power)
        print ("y cor of power is ", self.y_power)        
        print ("x_top    :", self.x_top,"   y_top  :",self.y_top)
        print ("x_bot   :", self.x_bot,"    y_bot  :",self.y_bot)
        print ("x_right :", self.x_right,"  y_right  :",self.y_right)
        print ("x_left   :", self.x_left,"  y_left :",self.y_left)
        print ("distance_CENTER:",self.distance_center)
        print ("distance_top:",self.distance_top)
        print ("distance_bot:",self.distance_bot)
        print ("distance_left:",self.distance_left)
        print ("distance_right:",self.distance_right)
    
    def point_distance(self,x1,y1,x2,y2):  
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return dist  

    def move_top(self):
        #print("move top:")
        self.agent.sety(self.agent.ycor() + self.agent.dy)
        #self.calculate_distance()

    def move_bot(self):
        #print("move bot:")
        self.agent.sety(self.agent.ycor() - self.agent.dy)
        #self.calculate_distance()

    def move_right(self):
        #print("move right:")
        self.agent.setx(self.agent.xcor() + self.agent.dx)
        #self.calculate_distance()

    def move_left(self):
        #print("move left:")
        self.agent.setx(self.agent.xcor() - self.agent.dx)
        #self.calculate_distance()
