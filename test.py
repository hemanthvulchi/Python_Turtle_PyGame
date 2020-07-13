#this file is to test code snippets
import turtle
import random

        
# self.speed(0)
# self.shape("cicle")
# self.color("black")
# self.penup()
# self.goto(random.randrange(-390,390),random.randrange(-390,390))
internal=10
print ("init value is ", internal)
print ("test2 value is:" , internal)


def testfun(value):
    value = 50
    print("value in ",value)

a=10
print("a in",a)
testfun(a)
print("a out",a)

def try_to_change_list_contents(the_list):
    print('got', the_list)
    the_list.append('four')
    print('changed to', the_list)

outer_list = ['one', 'two', 'three']

print('before, outer_list =', outer_list)
try_to_change_list_contents(outer_list)
print('after, outer_list =', outer_list)