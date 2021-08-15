# TrackingExpert+
TrackingExpert+ introduces a set of tools for augmented reality (AR) application development. In particular, it provides for 6-degree-of-freedom (DoF) pose measurement and 3D visualization capabilities. The focus is on industrial AR applications for tasks such as assembly training and task-support, maintenance, and others. 

![Title figure](https://github.com/rafael-radkowski/TrackingExpertPlus/blob/master/data/media/Tracking_expert_V1.png)

The software allows a user to detect objects and track their 6-DoF pose using point cloud data acquired from an RGB-D camera. The underlying pose estimation approach is model-based: it requires a 3D model of an object of interest as input data. Underneath the hood, TrackingExpert+ relies on features descriptors and statistical pattern matching to perform its task. Its feature descriptor employs surface curvatures of an object and the relative curvature distribution around a point. This approach renders the software highly robust in the presence of noise, occlusion, and other obstacles since it can work a minimum visible surface. 

### Features
TrackingExpert+ provides the following features:
 * Point cloud generation from an RGB-D image.
 * Normal vector estimation.
 * Point cloud sampling.
 * Object detection in point clouds.
 * Object 6-DoF pose estimation.
 * AR assembly training example application. 

Additionally, a Unity plugin is under development, which allows a user to employ the HoloLens along with TrackingExpert+ for any AR application.

### Prerequisites
All code runs (tested) on Windows 10, Version 1803.
It requires the following 3rd party tools:
 * [OpenCV v3.4.5](https://opencv.org)
 * [Eigen3 v3.3.7](http://eigen.tuxfamily.org)
 * [GLEW v2.1.0](http://glew.sourceforge.net)
 * [GLFW v3.2.1](https://www.glfw.org)
 * [glm, v0.9.9.3](https://glm.g-truc.net/0.9.9/index.html)
 * [Cuda V 9.2](https://developer.nvidia.com/cuda-92-download-archive)

The code was prepared with Visual Studio 2019. Note that this is only the software the development team uses. 
All components should also work with never software veersion.  

### Installation Instructions and Manual
Follow the link to the [TrackingExpert+ Installation and User Manual](https://docs.google.com/document/d/1IpHlpnFFG5dZNQ4PCa8HabXDFIdaZcVf4GVDytQKIK8/edit?usp=sharing). Please leave comments if you think information is missing, incorrect, or incomplete. 
