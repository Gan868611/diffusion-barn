#!/bin/bash
for i in $(seq 1 1 49)
do
    echo `expr $i \* 6 ` # 50 test BARN worlds with equal spacing indices: [0, 6, 12, ..., 294]
    echo $n
        for j in $(seq 1 1 10)
        do            
            # run the test
            python3 run.py --world_idx $n
            sleep 5
        done
done