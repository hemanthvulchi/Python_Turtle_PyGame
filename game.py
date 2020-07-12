#my own pong game by @hemanthvulchi
#using tutorial by tokyoedtech
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
while True:
    world.worldscreen.update()
    ai_agent.move() 

