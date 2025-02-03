#!/bin/bash

ATTEMPT=1
success_count=0
failure_count=0
#every 5 env is the test env [0,5,10.....]
for j in $(seq 0 5 295); do
    for ((i=0; i<ATTEMPT; i++)); do
        echo "==== Running world $j ===="
        success=false
        nohup timeout 100s  python ./scripts/run_rviz_imit.py --world_idx $j > ./nohup_out/run_rviz_imit$j-try_$i.log 2>&1 &
        wait $!
        result=$?  # Capture the exit status of the python command
        if [ $result -eq 200 ]; then
            echo "==== Success ===="
            success=true
            success_count=$((success_count + 1))
            echo 
            echo "==== Killed gazebo and rviz, Sleeping ===="
            pkill -f  gzserver
            pkill -f rviz
            sleep 5
        else
            echo "Failed"
            failure_count=$((failure_count + 1))
            echo "==== Killed gazebo and rviz, Sleeping ===="
            pkill -f  gzserver
            pkill -f rviz
            sleep 5
        fi
        total_attempts=$((success_count + failure_count))
        echo "Success Count: $success_count / Total Attempts: $total_attempts"
    done
done

echo "Total Successes: $success_count"
echo "Total Failures: $failure_count"