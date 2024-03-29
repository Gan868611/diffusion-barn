# Docker

## Build from Dockerfile

```shell
docker build . -t barn:[name]

docker run --rm -dt --name barn \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	barn:[name]

docker push mldarobotics/barn2024

```

## Tag and push to DockerHub

```
docker tag barn:[name] mldarobotics/barn2024:[name]
docker push mldarobotics/barn2024:[name]
```

## Pull from the DockerHub

```shell
docker run --rm -dt --name barn \
	--gpus all \
	-e DISPLAY=":0" \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v ~\Documents\GitHub\mlda-barn-2024\mlda_algo:/jackal_ws/src/mlda-barn-2024/mlda_algo \
	mldarobotics/barn2024:nmpcv1
```
