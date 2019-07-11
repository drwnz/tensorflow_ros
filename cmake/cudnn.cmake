include(FindPackageHandleStandardArgs)
unset(CUDNN_FOUND)

find_library(CUDNN_LIBRARY
  NAMES libcudnn cudnn
  HINTS /usr/local
        /usr/local/cuda
        ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64 lib/x86 targets/aarch64-linux
  )

find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY)

if(CUDNN_FOUND)
  set(CUDNN_LIBRARIES ${CUDNN_LIBRARY})
endif()
