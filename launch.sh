export SRC_DIR="$(pwd)/mlda_algo"
echo $SRC_DIR:/jackal_ws/src/mlda_algo

xhost +

if [ $# -eq 0 ]; then
    echo "Image: mldarobotics/barn2024:v1"
    IMAGE_NAME="mldarobotics/barn2024:v1"

else
    IMAGE_NAME="$1"
    echo "Image: $IMAGE_NAME"
fi

docker run --rm -dit --name barn \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $SRC_DIR:/jackal_ws/src/mlda-barn-2024/mlda_algo \
    -w /jackal_ws/src/mlda-barn-2024 \
	$IMAGE_NAME