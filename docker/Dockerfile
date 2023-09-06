# ros2 openvino toolkit env master f1b1ca4d914186a1881b87f103be9c6e910c9d80

ARG ROS_PRE_INSTALLED_PKG
FROM osrf/ros:${ROS_PRE_INSTALLED_PKG}
#FROM osrf/ros:humble-desktop-full
ARG VERSION

# setting proxy env --option
# If needed, enable the below ENV setting by correct proxies.
ENV http_proxy="http://child-prc.intel.com:913"
ENV https_proxy="http://child-prc.intel.com:913"
ENV no_proxy="127.0.0.1, localhost"
#ENV ftp_proxy="http://child-prc.intel.com:913"
#ENV https_proxy="http://child-prc.intel.com:913"


# default shell type
SHELL ["/bin/bash", "-c"]

# ignore the warning
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
RUN apt-get update && apt-get install --assume-yes apt-utils

# install openvino 2022.1
# https://docs.openvino.ai/2022.1/openvino_docs_install_guides_installing_openvino_apt.html
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -s https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo apt-key add -
RUN echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2023.list
RUN apt update && apt-cache search openvino && apt install -y openvino

# install librealsense2
#RUN apt-get install -y --no-install-recommends \
#software-properties-common
# https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
# Make sure you set http-proxy in below commands if your environment needs.
RUN mkdir -p /etc/apt/keyrings
RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | tee /etc/apt/keyrings/librealsense.pgp > /dev/null
RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
tee /etc/apt/sources.list.d/librealsense.list
RUN apt-get update
RUN apt-get install -y --no-install-recommends  \
librealsense2-dkms \
librealsense2-utils \
librealsense2-dev \
librealsense2-dbg \
libgflags-dev \
libboost-all-dev \
&& rm -rf /var/lib/apt/lists/*


# other dependencies
RUN apt-get update && apt-get install -y python3-pip && python3 -m pip install -U \
numpy \
networkx \
pyyaml \
requests \
&& apt-get install -y --no-install-recommends libboost-all-dev
WORKDIR /usr/lib/x86_64-linux-gnu
RUN pip install --upgrade pip

# build ros2 openvino toolkit
WORKDIR /root
RUN mkdir -p catkin_ws/src
WORKDIR /root/catkin_ws/src
RUN git clone https://github.com/intel/ros2_object_msgs.git \
&& git clone -b ros2 https://github.com/intel/ros2_openvino_toolkit.git
RUN apt-get install ros-${VERSION}-diagnostic-updater
WORKDIR /root/catkin_ws
RUN source /opt/ros/${VERSION}/setup.bash  && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release


#install depends deb&&python  packages
RUN apt-get install -y python3-colcon-common-extensions \
  ros-humble-diagnostic-updater

RUN apt-get update

