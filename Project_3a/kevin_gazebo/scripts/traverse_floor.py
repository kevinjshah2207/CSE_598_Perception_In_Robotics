#!/usr/bin/env python

from turtlebot3_controller import turtlebot3_controller
# import rotate as initial


def moveAround(controller):
    
    # 90 points towards +y 
    controller.goToPoint(7.0,0.0,180)
    controller.goToPoint(-3.5,0.0,0)
    # controller.goToPoint(-3.5,0.0,0)
    # controller.goToPoint(-3,2.75,-90)
    # controller.goToPoint(-3.0,0.0,0)
    controller.goToPoint(6.0,0.0,90)



if __name__ == '__main__':
    controller = turtlebot3_controller()
    moveAround(controller)
    controller.make_initial()
    controller.shutdown()