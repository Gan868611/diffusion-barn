#!/bin/sh
singularity exec -i --nv -n --network=none -p -B `pwd`:/jackal_ws/src/nav-competition-icra2022 ${1} /bin/bash ./singularity_query.sh
