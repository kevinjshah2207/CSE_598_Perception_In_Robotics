#!/usr/bin/env python

'''
	Please download the ssd_mobilenet_v2_coco_2018_03_29 library for this program to work and place it in 
	How to run this file:
	1. cd ~/home/kevinjshah2207/catkin_ws/src/pursuit_evasion/
	2. source devel/setup.bash
	3. roslaunch pursuit_evasion robot_amcl.launch map_file:=/home/kevinjshah2207/catkin_ws/src/pursuit_evasion/maps/map1.yaml
	4. rosrun pursuit_evasion person_follower.py
	5. roslaunch pursuit_evasion move_evader.launch world_index:=0
'''


from __future__ import print_function


import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge, CvBridgeError

MODELS_PATH = '/home/kevinjshah2207/catkin_ws/src/pursuit_evasion/models'

import os
import sys
import numpy as np
import tensorflow as tf


sys.path.append(os.path.join(MODELS_PATH, 'research'))
sys.path.append(os.path.join(MODELS_PATH, 'research/slim'))

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class person_follower:

	def __init__(self):
		print('ROS_OpenCV_bridge has been initialized')
		self.image_pub = rospy.Publisher("/CV_image", Image, queue_size=5)
		self.cmd_vel_pub = rospy.Publisher("/tb3_0/cmd_vel", Twist, queue_size=5)
		self.bridge = CvBridge()

		model = tf.saved_model.load(os.path.join(MODELS_PATH, 'ssd_mobilenet_v2_coco_2018_03_29/saved_model/'))
		self.detection_model = model.signatures['serving_default']
		print('Vision Model has been Loaded')

		utils_ops.tf = tf.compat.v1

		tf.gfile = tf.io.gfile


		PATH_TO_LABELS = os.path.join(MODELS_PATH, 'research/object_detection/data/mscoco_label_map.pbtxt')
		self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


		image_sub = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, self.callback)

		print("Subscribed to image_raw")


	def callback(self, ros_image):
		print('callback has been initiated')
		try:
			cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
		except CvBridgeError as e:
			print(e)

		output_dict = self.run_inference_for_single_image(cv_image)
		self.draw_output(cv_image, output_dict)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
		except CvBridgeError as e:
			print(e)

		self.follow_person(output_dict)

	def run_inference_for_single_image(self, image):
		image = np.asarray(image)
		input_tensor = tf.convert_to_tensor(image)
		input_tensor = input_tensor[tf.newaxis, ...]

		output_dict = self.detection_model(input_tensor)

		num_detections = int(output_dict.pop('num_detections'))
		output_dict = {key: value[0, :num_detections].numpy()
						for key, value in output_dict.items()}
		output_dict['num_detections'] = num_detections

		output_dict['detection_classes'] = output_dict['detection_classes'].astype(
			np.int64)

		if 'detection_masks' in output_dict:
			detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					output_dict['detection_masks'], output_dict['detection_boxes'],
					image.shape[0], image.shape[1])
			detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
											tf.uint8)
			output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

		return output_dict

	def draw_output(self, np_img, output_dict):
		vis_util.visualize_boxes_and_labels_on_image_array(
			np_img,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			self.category_index,
			instance_masks=output_dict.get('detection_masks_reframed', None),
			use_normalized_coordinates=True,
			line_thickness=8)

	
	def follow_person(self, output_dict):
		'''
			only publish when we find a person
		'''
		move_cmd = Twist()	

		index = -1
		for i in range(len(output_dict['detection_classes'])):
			if (output_dict['detection_classes'][i] == 1 and output_dict['detection_scores'][i] > 0.4):
				index = i
				break		

		if (index > -1):
			box = output_dict['detection_boxes'][index]
			box_center = (box[1] + box[3])/2.0

			move_cmd.angular.z = -1.1 * (box_center - 0.5)
			move_cmd.linear.x = 0.21

			debug = False
			debug = True
			if(debug):
				print('box_center: ', box_center)
				print('angular.z: {:.2f}'.format(move_cmd.angular.z))
			self.cmd_vel_pub.publish(move_cmd)

				

def main(args):
	rospy.init_node('person_follower', anonymous=False) 
	pf = person_follower()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
