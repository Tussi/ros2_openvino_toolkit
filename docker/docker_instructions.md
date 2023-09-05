# Run Docker Images For ROS2_OpenVINO_Toolkit

**NOTE:**
Below steps have been tested on **Ubuntu 22.04**.

## 1. Environment Setup
* System with Ubuntu20.04 or 22.04 installed  
* Realsense Camera inserted
* Dockerfile(docker/ros2_2021.4/ros2_foxy/Dockerfile)
* (Optional)Converted models(gearbolt-model: intel@10.239.89.24:/home/intel/ros2_ov/gearbolt-model)

## 2. Install Docker
```
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
	"deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
	  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## 3. add docker Proxy
```
sudo mkdir -p /etc/systemd/system/docker.service.d/
		 
sudo vim /etc/systemd/system/docker.service.d/proxy.conf
[Service]
Environment="HTTPS_PROXY=http://child-prc.intel.com:913"
		 
sudo vim /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://child-prc.intel.com:913" "HTTPS_PROXY=http://child-prc.intel.com:913"
		 
sudo systemctl daemon-reload
sudo systemctl restart docker
```
## 4. Build image
```
sudo docker build --build-arg ROS_PRE_INSTALLED_PKG=foxy-desktop --build-arg VERSION=foxy --build-arg "HTTP_PROXY=http://proxy-prc.intel.com:913" -t openvino_demo .
```

## 5. Running the Demos
* Install dependency
```
  xhost +
```
* run docker image
```
  sudo docker images
  sudo docker run -itd --network=host --privileged -p 40080:80  -p 48080:8080 -p 8888:8888 -p 9000:9000 -p 9090:9090  --device /dev/dri -v /dev/:/dev/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix:0 -e AMR_TYPE=symg --name ros2_foxy_openvino_202201 openvino_demo bash
  sudo docker exec -it ros2_foxy_openvino_202201 bash
  ```
* In Docker Container

* Preparation
```
source /opt/ros/foxy/setup.bash
source install/setup.bash
mkdir -p  /opt/openvino_toolkit/models/convert/public/FP32/yolov8n
cp gearbolt-model /opt/openvino_toolkit/models/convert/public
```

* See all available models
```
omz_downloader --print_all
```

* Download the optimized Intermediate Representation (IR) of model (execute once), for example:
```
cd ~/catkin_ws/src/ros2_openvino_toolkit/data/model_list
omz_downloader --list download_model.lst -o /opt/openvino_toolkit/models/
```
*If the model (tensorflow, caffe, MXNet, ONNX, Kaldi) need to be converted to intermediate representation (such as the model for object detection):
```
cd ~/catkin_ws/src/ros2_openvino_toolkit/data/model_list
omz_converter --list convert_model.lst -d /opt/openvino_toolkit/models/ -o /opt/openvino_toolkit/models/convert
```
* YOLOV8
  ```
  mkdir -p yolov8 && cd yolov8
  pip install ultralytics
  apt install python3.10-venv
  python3 -m venv openvino_env
  source openvino_env/bin/activate
  python -m pip install --upgrade pip
  pip install openvino-dev	
  pip install openvino-dev[extras]
  pip install openvino-dev[tensorflow2,onnx]
  #yolo export model=yolov8n.pt format=openvino
  yolo export model=yolov8n.pt format=onnx opset=10
  mo --input_model yolov8n.onnx --use_legacy_frontend
  cp yolov8* /opt/openvino_toolkit/models/convert/public/FP32/yolov8n/
  ```

* copy label files (execute once)
```
sudo cp ~/catkin_ws/src/ros2_openvino_toolkit/data/labels/face_detection/face-detection-adas-0001.labels /opt/openvino_toolkit/models/intel/face-detection-adas-0001/FP32/
sudo cp ~/catkin_ws/src/ros2_openvino_toolkit/data/labels/face_detection/face-detection-adas-0001.labels /opt/openvino_toolkit/models/intel/face-detection-adas-0001/FP16/
sudo cp ~/catkin_ws/src/ros2_openvino_toolkit/data/labels/emotions-recognition/FP32/emotions-recognition-retail-0003.labels /opt/openvino_toolkit/models/intel/emotions-recognition-retail-0003/FP32/
sudo cp ~/catkin_ws/src/ros2_openvino_toolkit/data/labels/object_segmentation/frozen_inference_graph.labels /opt/openvino_toolkit/models/intel/semantic-segmentation-adas-0001/FP32/
sudo cp ~/catkin_ws/src/ros2_openvino_toolkit/data/labels/object_segmentation/frozen_inference_graph.labels /opt/openvino_toolkit/models/intel/semantic-segmentation-adas-0001/FP16/
sudo cp ~/catkin_ws/src/ros2_openvino_toolkit/data/labels/object_detection/vehicle-license-plate-detection-barrier-0106.labels /opt/openvino_toolkit/models/intel/vehicle-license-plate-detection-barrier-0106/FP32
```

* Check the parameter configuration in ros2_openvino_toolkit/sample/param/xxxx.yaml before lauching, make sure parameters such as model_path, label_path and input_path are set correctly. 
  
  * run object yolo sample code input from RealSenseCamera.
  ```
   vim /root/catkin_ws/install/openvino_node/share/openvino_node/param/pipeline_object_yolo.yaml
Pipelines:
- name: object
  inputs: [RealSenseCamera]
  #input_path: to/be/set/image_path
  infers:
- name: ObjectDetection
  model: /opt/openvino_toolkit/models/convert/public/FP32/yolov8n/yolov8n.xml
      model_type: yolov8 #yolov8
      engine: CPU #MYRIAD
      label: to/be/set/xxx.labels
      batch: 1
      confidence_threshold: 0.5
      enable_roi_constraint: true # set enable_roi_constraint to false if you don't want to make the inferred ROI (region of interest) constrained into the camera frame
  outputs: [ImageWindow, RosTopic, RViz]
  connects:
    - left: RealSenseCamera
      right: [ObjectDetection]
    - left: ObjectDetection
      right: [ImageWindow]
    - left: ObjectDetection
      right: [RosTopic]
    - left: ObjectDetection
      right: [RViz]

OpenvinoCommon:
  ```
  
  ```  
  ros2 launch openvino_node pipeline_object_yolo.launch.py
  ```
* run segmentation_instance sample code input from RealSenseCamera.
  ```
vim /root/catkin_ws/install/openvino_node/share/openvino_node/param/pipeline_segmentation_instance.yaml
  ```
Pipelines:
- name: segmentation
  inputs: [RealSenseCamera]
  infers:
    - name: ObjectSegmentationInstance
      # for Yolov8 Seg models -----------------
      model: /opt/openvino_toolkit/models/convert/public/gearbolt-model/best_openvino_model/best.xml
      model_type: yolo
      label: /opt/openvino_toolkit/models/convert/public/gearbolt-model/configs/gearbolt.labels
      # for maskrcnn inception resnet -----------------
      #model: /opt/openvino_toolkit/models/convert/public/mask_rcnn_inception_resnet_v2_atrous_coco/FP32/mask_rcnn_inception_resnet_v2_atrous_coco.xml
      #model_type: maskrcnn
      #label: /opt/openvino_toolkit/labels/object_segmentation/frozen_inference_graph.labels #for maskrcnn
      #----------------------
      engine: CPU #"HETERO:CPU,GPU," #"HETERO:CPU,GPU,MYRIAD"
      batch: 1
      confidence_threshold: 0.7
  outputs: [ImageWindow, RosTopic, RViz]
  connects:
    - left: RealSenseCamera
      right: [ObjectSegmentationInstance]
    - left: ObjectSegmentationInstance
      right: [ImageWindow]
    - left: ObjectSegmentationInstance
      right: [RosTopic]
    - left: ObjectSegmentationInstance
      right: [RViz]

Common:
  ```
  ```  
  ros2 launch openvino_node pipeline_ segmentation_instance.launch.py
  ```
* run face detection sample code input from StandardCamera.
  ```
  ros2 launch dynamic_vino_sample pipeline_people.launch.py
  ```
  * run person reidentification sample code input from StandardCamera.
  ```
  ros2 launch dynamic_vino_sample pipeline_reidentification.launch.py
  ```
  * run person face reidentification sample code input from RealSenseCamera.
  ```
  ros2 launch dynamic_vino_sample pipeline_face_reidentification.launch.py
  ```
  * run face detection sample code input from Image.
  ```
  ros2 launch dynamic_vino_sample pipeline_image.launch.py
  ```
  * run object segmentation sample code input from RealSenseCameraTopic.
  ```
  ros2 launch dynamic_vino_sample pipeline_segmentation.launch.py
  ```
  * run object segmentation sample code input from Image.
  ```
  ros2 launch dynamic_vino_sample pipeline_segmentation_image.launch.py
  ``` 
  * run vehicle detection sample code input from StandardCamera.
  ```
  ros2 launch dynamic_vino_sample pipeline_vehicle_detection.launch.py
  ```
  * run person attributes sample code input from StandardCamera.
  ```
  ros2 launch dynamic_vino_sample pipeline_person_attributes.launch.py
  ```

# More Information
* ROS2 OpenVINO discription writen in Chinese: https://mp.weixin.qq.com/s/BgG3RGauv5pmHzV_hkVAdw

###### *Any security issue should be reported using process at https://01.org/security*

