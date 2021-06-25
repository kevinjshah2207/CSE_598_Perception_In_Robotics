#!/usr/bin/env python3.8
import rospy
from geometry_msgs.msg import Twist

def draw_initial():
    #Create a new node(according to instructions /my_initials)
    rospy.init_node('my_initials', anonymous=True)
    velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()
    #Speed of Turtle
    PI = 3.1415
    speed = 1
    radius = 1
    distance = PI * radius * 1.2

#Creating the first half of S
    vel_msg.linear.x = -radius
    vel_msg.linear.y = 0.4
    vel_msg.angular.z = -speed/radius
    t0 = rospy.Time.now().to_sec()
    current_angle = 0
    current_distance = 0

    while(current_distance < distance):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_distance = speed*(t1-t0)

#Bringing Turtle back to original position
    vel_msg.linear.x = radius
    vel_msg.linear.y = -0.4
    vel_msg.angular.z = speed/radius
    t0 = rospy.Time.now().to_sec()
    current_angle = 0
    current_distance = 0

    while(current_distance < distance):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_distance = speed*(t1-t0)

#Creating Second Half of S
    
    vel_msg.linear.x = radius
    vel_msg.linear.y = -0.4
    vel_msg.angular.z = -speed/radius
    t0 = rospy.Time.now().to_sec()
    current_angle = 0
    current_distance = 0

    while(current_distance < distance+0.3):
        velocity_publisher.publish(vel_msg)
        t1 = rospy.Time.now().to_sec()
        current_distance = speed*(t1-t0)

#Bringing the Turtle to a stop

    vel_msg.linear.x = 0
    vel_msg.linear.y = 0.0
    vel_msg.angular.z = 0
    velocity_publisher.publish(vel_msg)
    rospy.spin()

if __name__ == '__main__':
    try:
        draw_initial()
    except rospy.ROSInterruptException: pass
