#my own ambient AI game by @hemanthvulchi
#uses distance approximation to guide the agent towards the powersource
#export DISPLAY=:1
#shoutout to @Chingling152/Python-Space-Invaders for teaching the basics of classes
#Part 1

import turtle
import worldobjects 
import powersource
import agent
import neuralnet_pytorch
print("Hello world turtle")


# Main game#
class Gameengine:
    
    # world: worldobjects.WorldScreen
    # psrc: powersource.PowerSource
    # ai_agent: agent.Agent

    def __init__(self):
        self.world = worldobjects.WorldScreen(800, 800)
        self.psrc = powersource.PowerSource(self.world)
        self.ai_agent = agent.Agent(self.world)

    def print_distances(self):
        self.ai_agent.calculate_distance(self.psrc.psource.xcor(), 
                                         self.psrc.psource.ycor())
        self.ai_agent.print_distance()

    def worldexit(self):
        turtle.Screen().bye()

    #listen to key strokes
    def worldlisten(self):
        self.world.worldscreen.listen()
        self.world.worldscreen.onkeypress(self.print_distances, "z")
        self.world.worldscreen.onkeypress(self.ai_agent.move_top, "w")
        self.world.worldscreen.onkeypress(self.ai_agent.move_bot, "s")
        self.world.worldscreen.onkeypress(self.ai_agent.move_right, "d")
        self.world.worldscreen.onkeypress(self.ai_agent.move_left, "a")
        self.world.worldscreen.onkeypress(self.worldexit, "p")

    # main loop
    def mainloop_test(self):
        while True:
            self.world.worldscreen.update()
            self.psrc.check_agent(self.ai_agent)
    
    # main loop
    def mainloop(self, game_ai):
        while True:
            self.world.worldscreen.update()
            
            # observe the world from an agent's perspective
            
                       
            
            # to move powersource, if it is beside the agent
            self.psrc.check_agent(self.ai_agent)


#main game 
while True:
    game_ai = NeuralNetPytorch()
    game_world = Gameengine()
    game_world.worldlisten()
    game_world.mainloop(game_ai)
    break
