# this contains the agent itself and the functions it can perform
import turtle
import math


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
        self.distance_top = self.point_distance(self.x_top, self.y_top, self.x_power, self.y_power)
        self.distance_bot = self.point_distance(self.x_bot, self.y_bot, self.x_power, self.y_power)
        self.distance_right = self.point_distance(self.x_right, self.y_right, self.x_power, self.y_power)
        self.distance_left = self.point_distance(self.x_left, self.y_left, self.x_power, self.y_power)
        self.distance_center = self.point_distance(self.agent.xcor(), self.agent.ycor(), self.x_power, self.y_power)        

    # calculating distance beteen the agent and the powersource
    def get_distance(self, x_pwr, y_pwr):
        # calculating eyes of the agent
        self.calculate_distance(x_pwr, y_pwr)
        distances = [self.distance_top, self.distance_bot, self.distance_right, self.distance_left]
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
        print("distance_CENTER:", self.distance_center)
        print("distance_top:", self.distance_top)
        print("distance_bot:", self.distance_bot)
        print("distance_left:", self.distance_left)
        print("distance_right:", self.distance_right)
    
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
            self.move_right()
        elif movement_direction == 3:
            self.move_left()

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
