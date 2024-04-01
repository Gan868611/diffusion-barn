# MLDA_EEE

This is the repository for the proposed solution of Team MLDA_EEE (Machine Learning & Data Analytics) from Nanyang Technological University, Singapore

The original readme file is in [README_BARN.md](./README_BARN.md)

# Navigation Stack

Launch file at `./jackal_helper/launch/move_base_mlda.launch`

# Docker Image

This is the docker image for the April 1st Soft-Deadline submission

```shell
docker pull mldarobotics/barn2024:april1
```

Start the docker container named `barn` in the background. Use VSCode `Dev Container` extension to connect to the running container

```shell
xhost+ # Allow connections to X server

docker run --rm -dt --name barn \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/mlda-barn-2024:/jackal_ws/src/mlda-barn-2024 \
  mldarobotics/barn2024:april1
```

In the docker, the folder structure is similar to the suggested folder structure in the original README_BARN.md

```shell
# Navigate to this folder but in the container
cd /jackal_ws/src/mlda-barn-2024/
python3 run.py --world_idx 0
```

# Singularity Image

We can build the Singularity image but it cannot execute the `run.py` program through it due to `GLIBC=2.34 missing` error and we cannot fix. In the `.def` file, we also installed all the necessary `ros-melodic-*` packages instead of relying on `rosdep`
