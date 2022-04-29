#!/usr/bin/env python3
from email.errors import ObsoleteHeaderDefect
from statistics import mode
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from move_robot import MoveTurtlebot3
from darknet_ros_msgs.msg import BoundingBoxes
from apriltag_ros.msg import AprilTagDetectionArray

class LineFollower(object):
    def __init__(self):
        self.flag = 0
        self.mode = 0 
        self.stop = 0 
        self.auto_nav_object = 0
        self.stop_once = 1
        self.apriltags = 0
        self.move = Twist()
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image",Image,self.camera_callback)
        self.stop_sign_subscriber = rospy.Subscriber('/darknet_ros/bounding_boxes' , BoundingBoxes, self.stop_callback)
        self.vel_pub = rospy.Publisher("/cmd_vel",Twist, queue_size=10)
        self.laser_sub = rospy.Subscriber("/scan",LaserScan, self.obstacle_avoidance_callback)
        self.tag_sub = rospy.Subscriber('/tag_detections', AprilTagDetectionArray, self.apriltag_callback)
        self.moveTurtlebot3_object = MoveTurtlebot3()

    def stop_callback(self, msg):
        self.stop = 0
        if msg.bounding_boxes[len(msg.bounding_boxes)- 1].id == 11:
            self.stop = 1

    def camera_callback(self, data):
        # We select bgr8 because its the OpneCV encoding by default
        cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")

        # We get image dimensions and crop the parts of the image we dont need
        height, width, channels = cv_image.shape
        crop_img = cv_image[int((height/2)+100):int((height/2)+120)][1:int(width)]

        # Convert from RGB to HSV
        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
        # Threshold the HSV image to get only yellow colors
        lower_yellow = np.array([20,100,100])
        upper_yellow = np.array([70,255,255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
        # Calculate centroid of the blob of binary image using ImageMoments
        m = cv2.moments(mask, False)
            
        try:
            cx, cy = m['m10']/m['m00'], m['m01']/m['m00']
            self.mode = 1 
        except ZeroDivisionError:
            cx, cy = height/2, width/2
            
        # Draw the centroid in the resultant image
        # cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 
        cv2.circle(mask,(int(cx), int(cy)), 7, (0,0,255),-1)
        cv2.imshow("Original", cv_image)
        cv2.imshow("MASK", mask)
        cv2.waitKey(1)

        # controller
        if self.mode==1 and self.apriltags==0:
            print('line following is running!')
            err_x = cx - width/2
            twist_object = Twist()
            twist_object.linear.x = 0.08
            twist_object.angular.z = -err_x/450
            if self.stop == 1:
                if self.stop_once == 1:
                    print('stop sign detected!')
                    rospy.sleep(3)
                    twist_object.linear.x = 0
                    twist_object.angular.z = 0
                    self.moveTurtlebot3_object.move_robot(twist_object)
                    rospy.sleep(3)
                    self.stop = 0
                    self.stop_once = 0

            # Make it start turning
            self.moveTurtlebot3_object.move_robot(twist_object)

    def clean_up(self):
        self.moveTurtlebot3_object.clean_class()
        cv2.destroyAllWindows()
 
    def obstacle_avoidance_callback(self, laserscan): # defining a wall following function

        """ For lateral control """
        laserscan = list(laserscan.ranges[0:360]) # storing LiDAR data 
        
        right = laserscan[-90:-20]
        right_dist = sum(right)/len(right) # average distance of obstacles on the right 
        left = laserscan[20:90]
        left_dist = sum(left)/len(left) # average distance of obstacles on the left 
        err_side = left_dist-right_dist # estimating the error for P-Controller
        
        """ For longitudnal control """

        # front_dist = min(min(i for i in laserscan[(len(laserscan)-20):len(laserscan)] if i>0), min(i for i in laserscan[0:20] if i>0)) # front distance 
        # if len(front_dist) == 0:
        #     front_dist = 5

        front_dist = []
        front_right = laserscan[-20:]
        front_left = laserscan[0:20]
        if(len(front_left)>0 and len(front_right)>0):
            for i in range(0,len(front_left)):
                if front_left[i] != 0:
                    front_dist.append(front_left[i])
            for i in range(0,len(front_right)):
                if front_right[i] != 0:
                    front_dist.append(front_right[i])

            front_dist = min(front_dist)

        err_front = front_dist-0.4 # setting desired distance to be 0.2 for sim -- 0.35 for real-world
        kp_side = 1
        kp_front = 0.5

        if self.mode==0 and not self.apriltags>0 and self.status==False:
            """ Desired angular and linear velocity """
            self.move.angular.z = np.clip(err_side*kp_side,-1.2, 1.2)
            self.move.linear.x  = np.clip(err_front*kp_front,-0.1,0.2) # max linear vel to 0.3 for sim -- 0.4 for real-world 

            print('obstacle avoidance is running!')
            # if move.linear.x < 0.01 and move.linear.x > 0:
            #     move.angular.z = 0.3
            self.vel_pub.publish(self.move)

    def apriltag_callback(self, tags):
        self.apriltags = len(tags.detections) 
        
        if self.apriltags>0:
            print('april tag detection is running!')
            
            self.x = tags.detections[0].pose.pose.pose.position.x
            self.z = tags.detections[0].pose.pose.pose.position.z
            linear_vel = 0.06
            angular_vel = 3
            #velocity controller 
            self.move.linear.x = self.z*linear_vel #desired linear velocity
            self.move.angular.z = self.x*angular_vel #desired angular velocity
            #publishing velocity
            self.vel_pub.publish(self.move)               
            # rate.sleep()

if __name__ == '__main__':
    rospy.init_node('autonomous_navigation', anonymous=True)
    rate = rospy.Rate(5)
    auto_nav_object = LineFollower()
    ctrl_c = False
    def shutdownhook():
        auto_nav_object.clean_up()
        rospy.loginfo("shutdown time!")
        ctrl_c = True
    rospy.on_shutdown(shutdownhook)
    while not ctrl_c:
        rate.sleep()
