export SRC_DIR="$(pwd)"
echo $SRC_DIR:/jackal_ws/src/mlda_algo


xhost +local:docker
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi


if [ $# -eq 0 ]; then
    # echo "Image: mldarobotics/barn2024:zhenyangv1"
    # IMAGE_NAME="mldarobotics/barn2024:zhenyangv1"
    echo "Image: mldarobotics/barn2024:zhenyangv2_cuda"
    IMAGE_NAME="mldarobotics/barn2024:zhenyangv2_cuda"

else
    IMAGE_NAME="$1"
    echo "Image: $IMAGE_NAME"
fi

echo "Display:$DISPLAY"
# export ROS_REMOTE_PC=$(ifconfig ens10f0 | awk '/inet / {print $2}') #replace enp70s0 with your network card
# export ROS_PORT=11311

docker run --rm -dit --name barn-v2 \
	--gpus all \
	-e DISPLAY=$DISPLAY \
	-e QT_X11_NO_MITSHM=1 \
	--env "XAUTHORITY=$XAUTH" \
    -v "$XAUTH:$XAUTH" \
	-e LIBGL_ALWAYS_SOFTWARE=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=all \
	--network=host \
    --privileged \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $SRC_DIR:/jackal_ws/src/mlda-barn-2024/ \
	$IMAGE_NAME

docker exec barn-v2 apt-get update
# docker exec barn apt-get install -y python-scipy
docker exec barn-v2 pip3 install diffusers hydra einops zarr
docker exec barn-v2 source ~/miniconda3/bin/activate
docker exec barn-v2 echo 'export PYTHONPATH=$PYTHONPATH:/jackal_ws/src/mlda-barn-2024/train_imitation/diffusion_policy' >> ~/.bashrc



