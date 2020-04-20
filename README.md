# Readme

## Getting started
Run following command in the root directory while a ```roscore``` already run's: 

```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

**Note:** In contrast to the given version the simulator call in /ros/src/styx/launch/server.launch as this lead to error's running ros in a VM while the simulator runs on the host. However, the simulator has to be started manually then.


## Architecture

The main architecture given in this project is kept here. However, slight adaptions have been made to improve perfomance. In the subsequent section these are explained in detail.

### Waypoint Loader
This module imports a reference path from a csv-file. The basic import of the waypoints is extended by introducing additional precalculated parameter in the corresponding *lane*-stucture, which is published for other ros-nodes. These additional parameters are mainly the yaw angle, the curvature and the reference acceleration in longitudinal direction. Though these parameters could be calculated during runtime it's more efficient to calculate them once during import to reduce the computational effort.

Additionally the velocity profile is limited to a maximum lateral and longitudinal acceleration. Therefore, the calculated curvature is used to limit the velocity in order to limit the lateral acceleration. Afterwards, the longitudinal acceleration of this velocity profile is reduced by iterating forward and backward through the profile. The velocity of the node ```v(k+1)``` or ```v(k-1)``` is constrained by ```[sqrt(v(k)*v(k) - 2*ax*ds(k)),sqrt(v(k)*v(k) + 2*ax*ds(k))]```, where ```ax``` represents the maximum longitudinal acceleration and ```ds``` the distance from node ```k``` to node ```k+1``` or ```k-1```. The resulting path is then published.

### Traffic Light Detection

The traffic light detection uses a camera image and a map to determine relevant traffic lights, especially if a red light is in range. The ```tl-classifier.py``` uses a image to determine wether there is a red traffic light or not using a deep-neural-network (details explained later). The image is undistorted first, if the image is provided by a real camera and not by a simulator. If there is a red light, it is matched with the next known traffic light in a map. The corresponding node is then published. However, the search for a traffic light is conducted only if the vehicle is near to a mapped traffic light.

#### Traffic Light Classifier

The traffic light classifer uses a *ssd_mobilenet_v2_coco* in order to determine the color of the traffic light. This network is trained using the Tensorflow Object Detection API (a good how-to is given by a fellow udacity student [here](https://github.com/marcomarasca/SDCND-Traffic-Light-Detection)).

I tested various networks, however i relied on a pretty small pretrained network with *ssd_mobilenet_v2_coco* as my laptop does not have a powerful GPU. Even this small network took about 250ms for evaluation localy. The final network was trained just using self-labeled simulator data and the provided data from carla, giving an accuracy of 95% on both. Thus, I used the same network for real-world and simulator data. I additionally tried to use data from the [LISA](https://www.kaggle.com/mbornoe/lisa-traffic-light-dataset) and [Bosch Dataset](https://github.com/bosch-ros-pkg/bstld), but these reduced significantly the performance in simulation. A larger net like *faster_rcnn_resnet101_coco* gave better results compared to *ssd_mobilenet_v2_coco* but there neglected due to the considerably longer runtime. However, even those results were worse in simulation than the model just trained the simulation and carla data. A further problem was that bosch-data has another aspect ratio than LISA and the udacity data, thus ssd-networks are problematic as they rely on a fixed size and different aspect ratios mapped on a fixed image size lead to distortions.

Importers for those datasets to a tensorflow record file can be found in the tools folder.

The detected traffic lights by the net are online verified with the map data e.g. detected traffic light and mapped traffic light have to be close to each other. Also the aspect ration of the traffic light have to be approximately 1/3. If the detection has also a sufficient rating by the neural network it is accepted and passed back to the tl_detector module.

**Note:** Though the net is trained to classify between red, green and yellow lights, it outputs everytime a red light if its unsure. Only if its sure there is a green light it allows for continue driving.

### Waypoint Updater
The waypoint updater basically manipulates the longitudinal velocity in order to stop at red traffic lights. The geometry of the path is maintained as given by the waypoint loader. The output of this module is just a short snippet in order to cover at least a few seconds e.g. to handle a timeout and to give the controller in the drive-by-wire system enough lookahead to compensate for latencies and to calculate smooth steering and powertrain commands.

### Drive-By-Wire

The drive-by-wire system basically consists of controller for lateral and longitudinal guidance of the vehicle to the reference path provided by the waypoint updater. The base implementation was the proposed feedforward controller for lateral guidance and PI-controller for throttle and P-controller for brake given in the Udacity Walktrough-Guide. It's setpoints - angular velocity and longitudinal velocity - were provided by the existing waypoint-following module

As this approach  did not lead to satisfying results, another path-controler is implemented directly in the drive-by-wire system processing the reference path directly. This part is located in the module ```path-controller.py```. First a localization on the reference path is conducted in order to obtain e.g. the cross-track-error, the orientation error or velocity error. 

Additionally parameters like a lookahead curvature as feedforward for the lateral control is calculated. To do so an estimated delay time is used to predict a lookahead distance. To get smooth results the curvature is averaged in the range [0,2*Prediction Distance]. This curvature is given together with the cross-track-error and orientation error in an LQR-controller. The underlying model is implemented in to way. On is a point mass with the curvature-error (curvature-setpoint - path-curvature) as input. The second implementation adds an additional integrator to the curvature-error realizing the curvature-setpoint-gradient as an input (path-controller.py lines:64-104). Both approaches perform well under various circumstances and way more precise than the preliminary implemented controller. For longitundinal guidance the original throttle and brake controller is used as they gave a sufficient performance. 

### Other Topics

#### Parklot Track
This track was modified by elimating the overlap of start and end, which avoids in certain situations that the car stops at the end of the track due to localization problems.

#### Simulating the Parklot
Another launch script is added to use the parklot-track in simulation: *launch/site_sim.launch*. However, in this configuration the camera properties are removed to avoid undistorting an already undistorted image. In addition the simulator bridge was added to communicate with the unity simulator.

### Further Open Points
* Ego Vehicle Pose Prediction
* Compensating for Image Recognition Latency
* Trajectory Generation in Waypoint Updater e.g. to get a drivable trajectory especially for the parking lot scenario

### Observed Problems
* Severe Timeouts between Simulator and VM up to 3 seconds


