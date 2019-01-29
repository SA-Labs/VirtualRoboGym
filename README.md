# VirtualRoboGym
We have tried to create a faster, safer and smarter method to train robots neural network using virtual simulation. 


In this we have used the robot HADRON, made a solidworks files and then exported it to URDF ROS format using SLDRT to URDF concerter and then we have used pybullet to create a virtual simulation and trained the robot virtually using deep q learning of KERAS and then currently I am working to test it in the real world.

To start the code

Step 1: install all the required liabrariries   [random, gym, numpy, collections import deque, keras, pybullet, math]

Step 2: correct the path of the urdf file in the folllowing pyhton script neural_net_for_hadron_1version2.py {hadron = p.loadURDF("ur location of urdf file")}

Step 3: run the script, create awsum stuff and stay tuned!!!!

will further update with he next versions of the codes and arduino codes for the robot 
pls do comment update

For further research paper and details check my blog: https://salabs.io
email: shresthagrawal.31@gmail.com / sunfall6@gmail.com


Credits to the creator of HADRON :https://www.thingiverse.com/thing:356398
for solid works to urdf converter:http://wiki.ros.org/sw_urdf_exporter
Keras:https://keras.io/
Pybullet: https://pypi.python.org/pypi/pybullet

kudos-----------------

