#ifndef GADGETRON_CONFIG_H
#define GADGETRON_CONFIG_H

#define GADGETRON_VERSION_MAJOR 4
#define GADGETRON_VERSION_MINOR 1
#define GADGETRON_VERSION_PATCH 1
#define GADGETRON_VERSION_STRING "4.1.1"
#define GADGETRON_CONFIG_PATH "share/gadgetron/config"
#define GADGETRON_PYTHON_PATH "share/gadgetron/python"
#define GADGETRON_GIT_SHA1_HASH "b400d168b32929f33c96c7114ffdf31d795aaa35"
#define GADGETRON_CUDA_NVCC_FLAGS " -arch=sm_50 -gencode arch=compute_50,code=sm_50 "
#define GADGETRON_VAR_DIR "/var/lib/gadgetron"

#endif //GADGETRON_CONFIG_H
