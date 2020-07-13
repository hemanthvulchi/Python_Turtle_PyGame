#my own ambient AI game by @hemanthvulchi
#uses distance approximation to guide the agent towards the powersource
#Part 1

import turtle
import worldobjects 
import powersource
import agent
print("Hello world turtle")


#Main game



world = worldobjects.WorldScreen(800,800)
psrc=powersource.PowerSource()
ai_agent=agent.Agent()


def testprint():
    ai_agent.calculate_distance(psrc)
    ai_agent.print_distance()

def worldexit():
    turtle.Screen().bye()

#listen to key strokes
world.worldscreen.listen()
world.worldscreen.onkeypress(testprint,"z")
world.worldscreen.onkeypress(ai_agent.move_top,"w")
world.worldscreen.onkeypress(ai_agent.move_bot,"s")
world.worldscreen.onkeypress(ai_agent.move_right,"d")
world.worldscreen.onkeypress(ai_agent.move_left,"a")
world.worldscreen.onkeypress(worldexit,"p")

#main loop
while True:
    world.worldscreen.update()


