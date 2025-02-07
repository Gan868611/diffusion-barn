#!/bin/bash

SINGLE_ENV_TRAIN_SIZE=3
#world 254 and 282 always fail
for j in $(seq 283 299); do
    for ((i=0; i<SINGLE_ENV_TRAIN_SIZE; i++)); do
        echo "==== Running world $j ===="
        success=false
        while [ "$success" = false ]; do
            nohup timeout 100s  python ./scripts/run_rviz_kul.py --world_idx $j \
                --inspection_data_output_filename kul_data_10Hz_v2.csv \
                > ./nohup_out/run_rviz_kul_10hz_$j-try_$i.log 2>&1 &
            wait $!
            result=$?  # Capture the exit status of the python command
            if [ $result -eq 200 ]; then
                echo "==== Success ===="
                success=true
                echo "==== Killed gazebo and rviz, Sleeping ===="
                pkill -f  gzserver
                pkill -f rviz
                sleep 7
            else
                echo "Fail... retrying"
                echo "==== Killed gazebo and rviz, Sleeping ===="
                pkill -f  gzserver
                pkill -f rviz
                sleep 7
            fi
        done
    done
done