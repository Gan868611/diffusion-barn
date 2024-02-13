# Shell
```shell
python3 /home/mlda/jackal_ws/src/mlda-barn-2024/run.py --world_idx 0 --gui --type eband
```

# Docker
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
```
