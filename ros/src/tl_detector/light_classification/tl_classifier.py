from styx_msgs.msg import TrafficLight

import numpy as np
import os
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
import time

class TLClassifier(object):
    def __init__(self):


        self.current_light = TrafficLight.UNKNOWN  # Default value if pass on network / or nothing detected.

        cwd = os.path.dirname(os.path.realpath(__file__))

        # Default to Simulation with 10 regions.
        CKPT = cwd+'/model/graph.pb'


        # 14, as pretrained on bosch dataset with 14 classes
        NUM_CLASSES = 3


        # https://github.com/tensorflow/tensorflow/issues/6698
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # end

        ##### Build network
        self.image_np_deep = None
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph, config=config)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.boxes_tensor = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.detections_tensor = self.detection_graph.get_tensor_by_name('num_detections:0')
        print("Loaded graph")


    def get_classification(self, image,assumed_distance,camera_info):
        """Determines the color of the traffic light in the image
        WORK IN PROGRESS
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        run_network = True  # flag to disable running network if desired

        if run_network is True:
            image_np_expanded = np.expand_dims(image, axis=0)

            #time0 = time.time()

            #detection
            with self.detection_graph.as_default():
                (num, boxes, scores, classes) = self.sess.run(
                        [self.detections_tensor, self.boxes_tensor, self.scores_tensor, self.classes_tensor]    ,
                        feed_dict={self.image_tensor: image_np_expanded})


            #time1 = time.time()

            #print("Time in milliseconds", (time1 - time0) * 1000)

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            size_tl_h = 1.0
            size_tl_w = 0.3

            size_img_px_per_m = 170.0 # pixels in X meter represent 1m
            size_img_x = 20.0 # X value for above parameter

            ar_tl = size_tl_h/size_tl_w
            # Traffic light thing
            self.current_light = TrafficLight.RED
            min_score_thresh = .65
            est_dist_eff = -1
            for i in range(num):
                if scores[i] > min_score_thresh:

                    # get estimated widt
                    est_width_img  = max(1e-10,(boxes[i][3] - boxes[i][1]) * float(camera_info['image_width'])/size_img_px_per_m)
                    est_height_img = max(1e-10,(boxes[i][2] - boxes[i][0]) * float(camera_info['image_height'])/size_img_px_per_m)

                    # focal_length
                    # ininite focal length for simulation
                    est_width_foc = -1/(1000.0/camera_info['focal_length_x'] - 1.0/est_width_img)
                    est_height_foc = -1/(1000.0/camera_info['focal_length_y'] - 1.0/est_height_img)
                    #est_width_foc = est_width_img
                    #est_height_foc = est_height_img

                    # switch lights if aspect_ratio_does not fit
                    if est_width_foc > est_height_foc:
                        temp = est_height_foc
                        est_height_foc = est_width_foc
                        est_width_foc = temp

                    #check aspect ratio
                    aspect_ratio = est_height_foc/est_width_foc
                    if aspect_ratio < 0.5*ar_tl or aspect_ratio > 2.0*ar_tl: # aspect ratio not matches
                        continue

                    # get_position
                    est_dist = 0.5*(size_tl_w/est_width_foc + size_tl_h/est_height_foc)*size_img_x

                    #check if detection an map matches, more dist to vehicle as traffic lights are on opposite side of crossing
                    if est_dist > assumed_distance + 40 or est_dist < assumed_distance - 20:
                        continue

                    # ok, all tests passed
                    # for simplification if its not green we don't drive
                    if classes[i] == 1:
                        self.current_light = TrafficLight.GREEN
                    else:
                        self.current_light = TrafficLight.RED

                    min_score_thresh = scores[i]
                    est_dist_eff = est_dist


        #print("Type:{}; Score: {}; Dist: {}; AssumedDist: {}".format(self.current_light,min_score_thresh,est_dist_eff,assumed_distance))

        return self.current_light
