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
        CKPT = cwd+'/graphs/ssd_mobilenet_v2_fine_ud/frozen_inference_graph.pb'


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


    def get_classification(self, image):
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

            time0 = time.time()

            # Actual detection.
            with self.detection_graph.as_default():  
                (num, boxes, scores, classes) = self.sess.run(
                        [self.detections_tensor, self.boxes_tensor, self.scores_tensor, self.classes_tensor]    ,
                        feed_dict={self.image_tensor: image_np_expanded})               


            time1 = time.time()

            #print("Time in milliseconds", (time1 - time0) * 1000) 
            
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            # Traffic light thing
            self.current_light = TrafficLight.UNKNOWN
            min_score_thresh = .50
            for i in range(num):
                if scores[i] > min_score_thresh:
                    
                    if classes[i] == 1:
                        self.current_light = TrafficLight.GREEN
                    elif classes[i] == 2:
                        self.current_light = TrafficLight.YELLOW
                    elif classes[i] == 3:
                        self.current_light = TrafficLight.RED
                        
                    min_score_thresh = scores[i]

        
       # print("{} and {}".format(self.current_light,min_score_thresh))
       
        return self.current_light