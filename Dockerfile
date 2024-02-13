FROM ros:melodic

RUN mkdir -p /jackal_ws/src
WORKDIR /jackal_ws/src

RUN apt install git

RUN git clone https://github.com/Daffan/nav-competition-icra2022.git
RUN git clone https://github.com/jackal/jackal.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel
RUN git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel
RUN git clone https://github.com/utexas-bwi/eband_local_planner.git

RUN apt-get update && apt-get install -y python3-pip build-essential
RUN pip3 install defusedxml rospkg netifaces numpy

# RUN source /opt/ros/melodic/setup.bash
RUN apt-get install -y ros-melodic-desktop-full ros-melodic-gmapping \
ros-melodic-robot-localization ros-melodic-joint-state-publisher-gui ros-melodic-navigation \
ros-melodic-hector-gazebo-plugins ros-melodic-velodyne-description ros-melodic-rosdoc-lite \
ros-melodic-twist-mux ros-melodic-sick-tim ros-melodic-teleop-twist-joy ros-melodic-pointgrey-camera-description \
ros-melodic-interactive-marker-twist-server ros-melodic-lms1xx

WORKDIR /jackal_ws

# Add start script
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN echo "source /jackal_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "alias python=python3" >> ~/.bashrc

