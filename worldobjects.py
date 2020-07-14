#this contains world information
import turtle

class WorldScreen:
    worldscreen: turtle.Screen()
    size_x = 0
    size_y = 0

    def __init__(self,x_size,y_size):
        self.size_x = (x_size/2) - 10
        self.size_y = (y_size/2) - 10
        self.worldscreen = turtle.Screen()
        self.worldscreen.title("Ambient agent @Vulchi")
        self.worldscreen.bgcolor("green")
        self.worldscreen.setup(width=x_size, height=y_size,startx=-400,starty=50)
        self.worldscreen.tracer(0)
    
    def worldexit(self):
        self.worldscreen.bye()




