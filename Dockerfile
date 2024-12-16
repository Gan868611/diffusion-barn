FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04 as cuda-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Add the CUDA environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN nvcc --version

# Installing ROS-melodic
RUN apt-get install -y gnupg2
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y lsb-core
ARG DEBIAN_FRONTEND=noninteractive


# INSTALL OTHER NECESSARY PACKAGES
RUN apt-get install -y vim
RUN apt-get install -y wget
RUN apt-get update
RUN apt-get install -y python-pip
RUN apt-get install -y libpng16-16
RUN apt-get install -y libjpeg-turbo8
RUN apt-get install -y libtiff5


RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y curl lsb-release ca-certificates dirmngr
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub


# # ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s -k https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y -qq ros-melodic-desktop-full

RUN apt-get install -y python-rosdep
# RUN rosdep init
# RUN rosdep update
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc


RUN mkdir -p /jackal_ws/src
WORKDIR /jackal_ws/src

RUN apt install -y git


RUN git clone https://github.com/jackal/jackal.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel
RUN git clone https://github.com/JvThunder/mlda-barn-2024.git
# RUN git clone https://gitlab.kuleuven.be/u0144428/free_space_motion_tube.git --branch barn2023

WORKDIR /jackal_ws/src/mlda-barn-2024/free_space_motion_tube
RUN mkdir -p build
WORKDIR /jackal_ws/src/mlda-barn-2024/free_space_motion_tube/build
RUN cmake ..
RUN make -j8 && make install

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh
RUN source ~/miniconda3/bin/activate


RUN apt-get update && apt-get install -y python3-pip build-essential
RUN pip3 install --upgrade pip
RUN pip3 install defusedxml rospkg netifaces numpy jupyter scipy matplotlib casadi scikit-learn pandas easydict scipy
RUN pip3 install --trusted-host pypi.org --trusted-host pytorch.org --trusted-host download.pytorch.org --trusted-host files.pypi.org --trusted-host files.pytorch.org torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
# Install ROS components
RUN apt-get install -y ros-melodic-desktop-full ros-melodic-gmapping \
    ros-melodic-robot-localization ros-melodic-joint-state-publisher-gui ros-melodic-navigation \
    ros-melodic-hector-gazebo-plugins ros-melodic-velodyne-description ros-melodic-rosdoc-lite \
    ros-melodic-twist-mux ros-melodic-sick-tim ros-melodic-teleop-twist-joy ros-melodic-teleop-twist-keyboard ros-melodic-pointgrey-camera-description \
    ros-melodic-interactive-marker-twist-server ros-melodic-lms1xx ros-melodic-laser-pipeline ros-melodic-controller-manager \ 
    python-scipy

WORKDIR /jackal_ws

# Add start script
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN echo "if [ ! -d /jackal_ws/devel ]; then ">> ~/.bashrc
RUN echo "  cd /jackal_ws" >> ~/.bashrc
RUN echo "  catkin_make" >> ~/.bashrc
RUN echo "else" >> ~/.bashrc
RUN echo "  source /jackal_ws/devel/setup.bash;" >> ~/.bashrc
RUN echo "fi" >> ~/.bashrc
RUN echo "alias python=python3" >> ~/.bashrc
RUN echo 'cd /jackal_ws/src/mlda-barn-2024' >> ~/.bashrc
WORKDIR /jackal_ws/src/mlda-barn-2024