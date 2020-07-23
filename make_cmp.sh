#!/bin/bash

echo "Build uses multiple threads. Please ensure you won't restrict other people's work."
echo "Detecting all jobs on current node:"
qstat -u "*"
echo -n "Is the queue empty (except for your own jobs)? [yes/no]: "
read PROCEED
if [[ "$PROCEED" != "yes" ]]; then
    exit 0
fi

echo -n "Have you checked htop for running interactive jobs? [yes/no]: "
read PROCEED
if [[ "$PROCEED" != "yes" ]]; then
    exit 0
fi

# load dependencies
module load CeresSolver/1.14.0-fosscuda-2018b

# build PoseLib dependency
cd ext/PoseLib
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../installed
make install -j16
cd ../../..
# build MultiCameraPose
mkdir build
cd build
cmake -DCERES_PATH=/home.others/eb/easybuild/software/CeresSolver/1.14.0-fosscuda-2018b ..
make -j4