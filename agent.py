# this contains the agent itself and the functions it can perform
import turtle
import math
from math import cos, sin, pi


class Agent():
    radius = 20

    #
    def __init__(self, screen_world):
        self.agent = turtle.Turtle()
        self.agent.speed(0)
        self.agent.shape("circle")
        self.agent.shapesize(stretch_wid=2, stretch_len=2)
        self.agent.color("black")
        self.agent.penup()
        # self.agent.goto(random.randrange(-390,390),random.randrange(-390,390))
        self.agent.goto(0, 0)
        self.agent.speed(0.05)
        self.agent.dx = 1
        self.agent.dy = 1
        self.size_x = screen_world.size_x
        self.size_y = screen_world.size_y

    # calculating distance beteen the agent and the powersource
    def calculate_distance(self, x_pwr, y_pwr):
        # calculating eyes of the agent
        self.x_top = self.agent.xcor()
        self.y_top = self.agent.ycor() + self.radius
        self.x_bot = self.agent.xcor()
        self.y_bot = self.agent.ycor() - self.radius
        self.x_right = self.agent.xcor() + self.radius
        self.y_right = self.agent.ycor()
        self.x_left = self.agent.xcor() - self.radius
        self.y_left = self.agent.ycor()
        # getting position of the powersource
        self.x_power = x_pwr
        self.y_power = y_pwr
        # calculating distance from eyes to the powersource
        self.distance_0center = self.point_distance(self.agent.xcor(), self.agent.ycor(), self.x_power, self.y_power)
        self.distance_3right = self.point_distance(self.x_right, self.y_right, self.x_power, self.y_power)
        self.distance_6bot = self.point_distance(self.x_bot, self.y_bot, self.x_power, self.y_power)
        self.distance_9left = self.point_distance(self.x_left, self.y_left, self.x_power, self.y_power)
        self.distance_12top = self.point_distance(self.x_top, self.y_top, self.x_power, self.y_power)

    # calculating distance beteen the agent and the powersource
    def get_distance(self, x_pwr, y_pwr):
        # calculating eyes of the agent
        #self.calculate_distance(x_pwr, y_pwr)
        distances = [self.distance_12top, self.distance_6bot, self.distance_9left, self.distance_3right]
        return distances

    # used to print current distance calculations for debugging purposes
    def print_distance(self):
        print("================")
        print("x cor of agent is ", self.agent.xcor())
        print("y cor of agent is ", self.agent.ycor())
        print("x cor of power is ", self.x_power)
        print("y cor of power is ", self.y_power)        
        print("x_top    :", self.x_top, "   y_top  :", self.y_top)
        print("x_bot   :", self.x_bot, "    y_bot  :", self.y_bot)
        print("x_right :", self.x_right, "  y_right  :", self.y_right)
        print("x_left   :", self.x_left, "  y_left :", self.y_left)
        print("distance_CENTER:", self.distance_0center)
        print("distance_top:", self.distance_12top)
        print("distance_bot:", self.distance_6bot)
        print("distance_left:", self.distance_9left)
        print("distance_right:", self.distance_3right)

    # calculating distance beteen the agent and the powersource
    def calculate_distances(self, x_pwr, y_pwr):
        # calculating eyes of the agent
        self.x_points = [0] * 12
        self.y_points = [0] * 12
        self.pdistance = [0] * 12
        self.x_top = self.agent.xcor()
        self.y_top = self.agent.ycor() + self.radius
        self.x_bot = self.agent.xcor()
        self.y_bot = self.agent.ycor() - self.radius
        self.x_right = self.agent.xcor() + self.radius
        self.y_right = self.agent.ycor()
        self.x_left = self.agent.xcor() - self.radius
        self.y_left = self.agent.ycor()
        # getting position of the powersource
        self.x_power = x_pwr
        self.y_power = y_pwr
        # calculating distance from eyes to the powersource
        self.distance_0center = self.point_distance(self.agent.xcor(), self.agent.ycor(), self.x_power, self.y_power)
        self.distance_3right = self.point_distance(self.x_right, self.y_right, self.x_power, self.y_power)
        self.distance_6bot = self.point_distance(self.x_bot, self.y_bot, self.x_power, self.y_power)
        self.distance_9left = self.point_distance(self.x_left, self.y_left, self.x_power, self.y_power)
        self.distance_12top = self.point_distance(self.x_top, self.y_top, self.x_power, self.y_power)

        angle = 0
        i = 0
        for i in range(0, 12):
            self.x_points[i] = round(self.agent.xcor() + (self.radius * cos((pi)*(angle/180))), 0)
            self.y_points[i] = round(self.agent.ycor() + (self.radius * sin((pi)*(angle/180))), 0)
            self.pdistance[i] = self.point_distance(self.x_points[i], self.y_points[i], self.x_power, self.y_power)
            #print("angle:", angle, " x:", self.x_points[i], " y:", self.y_points[i])
            angle += 30

    # calculating distance beteen the agent and the powersource
    def get_distances(self, x_pwr, y_pwr):
        # calculating eyes of the agent
        #self.calculate_distances(x_pwr, y_pwr)
        return self.pdistance

    # used to print current distance calculations for debugging purposes
    def print_distances(self):
        print("================")
        print("x cor of agent is ", self.agent.xcor())
        print("y cor of agent is ", self.agent.ycor())
        print("x cor of power is ", self.x_power)
        print("y cor of power is ", self.y_power)
        for i in range(0, 12):
            print("point:", i, " x:", self.x_points[i], " y:", self.y_points[i])
            print("distance:", self.pdistance[i])

    # math function to find the distance between two points
    def point_distance(self, x1, y1, x2, y2):
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist

    def move_agent(self, movement_direction):
        if movement_direction == 0:
            self.move_top()
        elif movement_direction == 1:
            self.move_bot()
        elif movement_direction == 2:
            self.move_left()
        elif movement_direction == 3:
            self.move_right()

    # function to move the agent to top / also included boundary checks
    def move_top(self):
        if self.agent.ycor() < self.size_y:
            self.agent.sety(self.agent.ycor() + self.agent.dy)

    # function to move the agent to bottom / also included boundary checks
    def move_bot(self):
        if self.agent.ycor() > -(self.size_y):
            self.agent.sety(self.agent.ycor() - self.agent.dy)

    # function to move the agent to right / also included boundary checks
    def move_right(self):
        if self.agent.xcor() < self.size_x:
            self.agent.setx(self.agent.xcor() + self.agent.dx)

    # function to move the agent to left / also included boundary checks
    def move_left(self):
        if self.agent.xcor() > -(self.size_x):
            self.agent.setx(self.agent.xcor() - self.agent.dx)

    def reset_agent(self):
        self.agent.goto(0, 0)
