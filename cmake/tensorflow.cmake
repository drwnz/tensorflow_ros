include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TENSORFLOW_INCLUDE_DIR
  NAMES tensorflow
  HINTS /usr/local/include/tensorflow /usr/include/tensorflow tensorflow/include/tensorflow
  )

find_library(TENSORFLOW_LIBRARY
  NAMES libtensorflow tensorflow
  HINTS /usr/lib /usr/local/lib tensorflow/lib
  )

find_package_handle_standard_args(TENSORFLOW DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)
find_package(CUDA)

if(NOT TENSORFLOW_FOUND)
  make_directory(${CMAKE_INSTALL_PREFIX}/tensorflow)
  if(CUDA_FOUND AND CUDNN_FOUND AND (${CUDA_VERSION} STREQUAL "9.0"))
    message(STATUS "Downloading GPU version of TensorFlow for CUDA 9.0")
    file(DOWNLOAD https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.12.0.tar.gz
      ${CMAKE_INSTALL_PREFIX}/tensorflow/tensorflow.tar.gz
      STATUS status
      )
  elseif(CUDA_FOUND AND CUDNN_FOUND AND (${CUDA_VERSION} STREQUAL "10.0"))
    message(STATUS "Downloading GPU version of TensorFlow for CUDA 10.0")
    file(DOWNLOAD https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz
      ${CMAKE_INSTALL_PREFIX}/tensorflow/tensorflow.tar.gz
      STATUS status
      )
  else()
    message(STATUS "Downloading CPU version of TensorFlow")
    file(DOWNLOAD https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.14.0.tar.gz
      ${CMAKE_INSTALL_PREFIX}/tensorflow/tensorflow.tar.gz
      STATUS status
      )
  endif()
  list(GET status 0 status_code)
  list(GET status 1 status_string)
  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "Error downloading TensorFlow: ${status_string}")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_INSTALL_PREFIX}/tensorflow/tensorflow.tar.gz WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/tensorflow)
  file(REMOVE ${CMAKE_INSTALL_PREFIX}/tensorflow/tensorflow.tar.gz)

  unset(TENSORFLOW_FOUND)
  set(TENSORFLOW_LIBRARY ${CMAKE_INSTALL_PREFIX}/tensorflow/lib/libtensorflow.so)
  set(TENSORFLOW_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/tensorflow/include)
  find_package_handle_standard_args(TENSORFLOW DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)
endif()

if(TENSORFLOW_FOUND)
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY})
  set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR})
endif()
