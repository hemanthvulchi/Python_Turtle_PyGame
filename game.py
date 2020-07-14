#my own ambient AI game by @hemanthvulchi
#uses distance approximation to guide the agent towards the powersource
#export DISPLAY=:1
#Part 1

import turtle
import worldobjects 
import powersource
import agent
print("Hello world turtle")


#Main game#
class Gameengine:
    
    # world: worldobjects.WorldScreen
    # psrc: powersource.PowerSource
    # ai_agent: agent.Agent

    def __init__(self):
        self.world = worldobjects.WorldScreen(800,800)
        self.psrc=powersource.PowerSource(self.world)
        self.ai_agent=agent.Agent(self.world)

    def testprint(self):
        self.ai_agent.calculate_distance(psrc)
        self.ai_agent.print_distance()

    def worldexit(self):
        turtle.Screen().bye()

    #listen to key strokes
    def worldlisten(self):
        self.world.worldscreen.listen()
        self.world.worldscreen.onkeypress(self.testprint,"z")
        #world.worldscreen.onkeypress(ai_agent.move_top(world),"w")
        self.world.worldscreen.onkeypress(self.ai_agent.move_top,"w")
        self.world.worldscreen.onkeypress(self.ai_agent.move_bot,"s")
        self.world.worldscreen.onkeypress(self.ai_agent.move_right,"d")
        self.world.worldscreen.onkeypress(self.ai_agent.move_left,"a")
        self.world.worldscreen.onkeypress(self.worldexit,"p")

    #main loop
    def mainloop(self):
        while True:
            self.world.worldscreen.update()
            self.psrc.check_agent(self.ai_agent)

#main game 
while True:
    game_ai = Gameengine()
    game_ai.worldlisten()
    game_ai.mainloop()
    break