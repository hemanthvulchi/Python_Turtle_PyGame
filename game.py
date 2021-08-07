#my own ambient AI game by @hemanthvulchi
#uses distance approximation to guide the agent towards the powersource
#export DISPLAY=:1
#shoutout to @Chingling152/Python-Space-Invaders for teaching the basics of classes
#Part 1

import turtle
import worldobjects 
import powersource
import agent
import neuralnet_pytorch as NeuralNetPytorch
import os
import random
import pongMath as pm

os.system('clear')
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
        self.ai_agent.calculate_distances(self.psrc.psource.xcor(),
                                          self.psrc.psource.ycor())
        self.ai_agent.print_distances()

    def worldexit(self):
        turtle.Screen().bye()

    def setRandomSkips(self, number):
        self.random_skips = number

    def changeRandomSkips1(self):
        self.setRandomSkips(1)
        print("Setting random skips:1")

    def changeRandomSkips2(self):
        self.setRandomSkips(2)
        print("Setting random skips:2")

    def changeRandomSkips5(self):
        self.setRandomSkips(5)
        print("Setting random skips:5")

    def changeRandomSkips9(self):
        self.setRandomSkips(9)
        print("Setting random skips:9")

    def changeRandomSkips10000(self):
        self.setRandomSkips(10000)
        print("Setting random skips:10000")

    def reset_agent(self):
        self.ai_agent.agent.goto(0, 0)

    # listen to key strokes
    def worldlisten(self):
        self.world.worldscreen.listen()
        self.world.worldscreen.onkeypress(self.print_distances, "z")
        self.world.worldscreen.onkeypress(self.ai_agent.move_top, "w")
        self.world.worldscreen.onkeypress(self.ai_agent.move_bot, "s")
        self.world.worldscreen.onkeypress(self.ai_agent.move_right, "d")
        self.world.worldscreen.onkeypress(self.ai_agent.move_left, "a")
        self.world.worldscreen.onkeypress(self.ai_agent.reset_agent, "r")
        self.world.worldscreen.onkeypress(self.worldexit, "p")
        self.world.worldscreen.onkeypress(self.changeRandomSkips1, "1")
        self.world.worldscreen.onkeypress(self.changeRandomSkips2, "2")
        self.world.worldscreen.onkeypress(self.changeRandomSkips5, "5")
        self.world.worldscreen.onkeypress(self.changeRandomSkips9, "9")
        self.world.worldscreen.onkeypress(self.changeRandomSkips10000, "0")

    # main loop
    def mainloop_test(self):
        while True:
            self.world.worldscreen.update()
            self.psrc.check_agent(self.ai_agent)
    
    # main loop
    def mainloop(self, game_ai):
        i = 0
        self.random_skips = 10000
        while True:
            # update the screen with the latest
            self.world.worldscreen.update()
            # calculate the agent distances from the power source
            self.ai_agent.calculate_distances(self.psrc.psource.xcor(),
                                              self.psrc.psource.ycor())
            agent_distances = self.ai_agent.get_distances(self.psrc.psource.xcor(),
                                                          self.psrc.psource.ycor())
            agent_y_actual_distances = self.ai_agent.get_distance(self.psrc.psource.xcor(),
                                                                  self.psrc.psource.ycor())                                                     
            # forward pass || agent also takes agent distances
            self.y_actual_index = pm.PongMath.minimum_index(agent_y_actual_distances, 4)
            self.y_index = game_ai.action_world(agent_distances)
            
            # this calculates the loss of the agent movement; i is just to print the iteration of the code
            self.loss = game_ai.observe_result(i, self.y_actual_index)

            if i % self.random_skips == 0:
                # this moves the agent in the resulting direction
                self.ai_agent.move_agent(self.y_index)
            else:
                self.ai_agent.move_agent(random.randint(0, 3))
                self.psrc.goto_random()


            # If aimed to move the agent through a procedural fashion
            #self.ai_agent.move_agent(self.y_actual_index)

            if i % 100 == 99:
                print("move towards", self.y_index)#

            # to move powersource, if it is beside the agent
            self.psrc.check_agent(self.ai_agent)
            i = i + 1


# main game 
while True:
    game_ai = NeuralNetPytorch.NeuralNetCustom()
    game_world = Gameengine()
    game_world.worldlisten()
    game_world.mainloop(game_ai)
    break
