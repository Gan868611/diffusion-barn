# Docker

## Build from Dockerfile

```shell
docker build . -t barn:icra --no-cache
```

## Tag and push to DockerHub

```shell
docker tag barn:icra mldarobotics/barn2024:icra
docker push mldarobotics/barn2024:icra
```

## Pull from the DockerHub

```shell
# run the docker and attach the volume of the docker to the local folder
docker run --rm -dt --name barn \
	--gpus all \
	-e DISPLAY="$DISPLAY" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ~/mlda-barn-2024:/jackal_ws/src/mlda-barn-2024 \
	mldarobotics/barn2024:icra

docker run --rm -dt --name barn \
	-e DISPLAY=":1" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	barn:icra

```

## ROS
- Clear map
```shell
rosservice call /move_base/clear_costmaps "{}"
```
