cmake_minimum_required(VERSION 2.8.3)
project(tensorflow_ros)

SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -g -Wall ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  )

include(cmake/cudnn.cmake)
include(cmake/tensorflow.cmake)

if(TENSORFLOW_FOUND AND CUDA_FOUND AND CUDNN_FOUND)
  if((${CUDA_VERSION} STREQUAL "9.0") OR (${CUDA_VERSION} STREQUAL "10.0"))
    if(${CUDA_VERSION} STREQUAL "9.0")
      set(TENSORFLOW_NEW_EXPERIMENTAL_API 0)
    else()
      set(TENSORFLOW_NEW_EXPERIMENTAL_API 1)
    endif()
    message(STATUS "-- USING TENSORFLOW IN GPU MODE --")
    message(STATUS "CUDA Version: " ${CUDA_VERSION})
    message(STATUS "CUDA Library: " ${CUDA_CUDA_LIBRARY})
    message(STATUS "CUDA Runtime: " ${CUDA_CUDART_LIBRARY})
    set(TENSORFLOW_CPU FALSE)
    catkin_package(
      INCLUDE_DIRS include
      LIBRARIES tensorflow_ros
      DEPENDS
        TENSORFLOW
        CUDA
        CUDNN
        roscpp
      )
  else()
    message(WARNING "The installed CUDA version " ${CUDA_VERSION} "is not compatible with TensorFlow")
  endif()
endif()
if(TENSORFLOW_FOUND AND ${TENSORFLOW_CPU})
  message(STATUS "-- USING TENSORFLOW IN CPU MODE --")
  set(TENSORFLOW_NEW_EXPERIMENTAL_API 1)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES tensorflow_ros
    DEPENDS
      TENSORFLOW
      roscpp
    )
endif()

if(TENSORFLOW_FOUND)
  message(STATUS "TensorFlow Library: " ${TENSORFLOW_LIBRARIES})
  add_definitions(-DTENSORFLOW_NEW_EXPERIMENTAL_API=${TENSORFLOW_NEW_EXPERIMENTAL_API})
  include_directories(tensorflow_ros
    include
    ${TENSORFLOW_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    )
  add_library(tensorflow_ros
    src/tensorflow_ros.cpp
    )
  target_link_libraries(tensorflow_ros
    ${TENSORFLOW_LIBRARIES}
    ${catkin_LIBRARIES}
    )
  add_dependencies(tensorflow_ros
    ${catkin_EXPORTED_TARGETS}
    )

  install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
    FILES_MATCHING PATTERN "*.hpp"
    )
  install(TARGETS tensorflow_ros
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )

  ### Unit Tests ###
  if(CATKIN_ENABLE_TESTING)
    find_package(rostest REQUIRED)
      add_rostest_gtest(test_tensorflow_ros
      test/test_tensorflow_ros.test
      test/src/test_tensorflow_ros.cpp
      )
    target_link_libraries(test_tensorflow_ros
      ${catkin_LIBRARIES}
      tensorflow_ros
      )
  endif()
else()
  message(FATAL_ERROR "'TensorFlow' is not installed. The library 'tensorflow_ros' will not be built.")
endif()
