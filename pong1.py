#my own pong game by @hemanthvulchi
#using tutorial by tokyoedtech
#Part 1

import turtle
print("Hello world turtle")


wn = turtle.Screen()
wn.title("Pong by @hemanth")
wn.bgcolor("green")
wn.setup(width=800, height=600,startx=-800,starty=50)
wn.tracer(0)

#score
scoreA=0
scoreB=0

#Paddle A
paddle_a=turtle.Turtle()
paddle_a.speed(0)
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5,stretch_len=1)
paddle_a.penup()
paddle_a.goto(-350,0)

#Paddle B
paddle_b=turtle.Turtle()
paddle_b.speed(0)
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5,stretch_len=1)
paddle_b.penup()
paddle_b.goto(+350,0)
paddle_b

#Ball
ball=turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0,0)
ball.dx = 0.2
ball.dy = 0.2

#pen
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0,260)
pen.write("Player A:0 Player B: 0", align="center", font=("Tahoma", 16,"normal"))


#function
def paddle_a_up():
    y=paddle_a.ycor()
    y = y + 30
    paddle_a.sety(y)

def paddle_a_down():
    y=paddle_a.ycor()
    y = y - 30
    paddle_a.sety(y)

def paddle_b_up():
    y=paddle_b.ycor()
    y = y + 30
    paddle_b.sety(y)

def paddle_b_down():
    y=paddle_b.ycor()
    y = y - 30
    paddle_b.sety(y)

#keyboard binding
wn.listen()
wn.onkeypress(paddle_a_up,"w")
wn.onkeypress(paddle_a_down,"s")
wn.onkeypress(paddle_b_up,"Up")
wn.onkeypress(paddle_b_down,"Down")

#Main game loop
while True:
    wn.update()

    #move the ball
    ball.setx(ball.xcor()+ball.dx)
    ball.sety(ball.ycor()+ball.dy)

    #border checking
    if ball.ycor() > 290:
       ball.sety(290)
       ball.dy = ball.dy * (-1)

    if ball.ycor() < -290:
       ball.sety(-290)
       ball.dy = ball.dy * (-1)

    if ball.xcor() > 390:
       ball.goto(0,0)
       ball.dx = ball.dx * -1
       scoreA= scoreA+1
       pen.clear()
       pen.write("Player A: {}  Player B: {}".format(scoreA,scoreB), align="center", font=("Tahoma", 16,"normal"))

    if ball.xcor() < -390:
       ball.goto(0,0)
       ball.dx = ball.dx * -1
       scoreB= scoreB+1
       pen.clear()
       pen.write("Player A: {}  Player B: {}".format(scoreA,scoreB), align="center", font=("Tahoma", 16,"normal"))

    # paddle wall collisions
    if (ball.xcor() > 340 and ball.xcor() < 350) and (ball.ycor() < paddle_b.ycor() + 40 and ball.ycor() > paddle_b.ycor() - 40):
       ball.setx(340)
       ball.dx = ball.dx * -1

    if (ball.xcor() < -340 and ball.xcor() > -350) and (ball.ycor() < paddle_a.ycor() + 40 and ball.ycor() > paddle_a.ycor() - 40):
       ball.setx(-340)
       ball.dx = ball.dx * -1