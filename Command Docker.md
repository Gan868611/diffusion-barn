
# Docker

## Build from Dockerfile
```shell
docker build . -t barn:latest

docker run --rm -dt --name barn \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	barn:latest

docker push mldarobotics/barn2024

```

## Tag and push to DockerHub

```
docker tag barn:kul mldarobotics/barn2024:kul
docker push mldarobotics/barn2024:kul
```


## Pull from the DockerHub
```shell
docker run --rm -dt --name barn \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	mldarobotics/barn2024:v1
```