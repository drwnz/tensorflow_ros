include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TENSORFLOW_INCLUDE_DIR
  NAMES tensorflow
  HINTS /usr/local/include/tensorflow /usr/include/tensorflow
  )

find_library(TENSORFLOW_LIBRARY
  NAMES libtensorflow tensorflow
  HINTS /usr/lib /usr/local/lib
  )

find_package_handle_standard_args(TENSORFLOW DEFAULT_MSG TENSORFLOW_INCLUDE_DIR TENSORFLOW_LIBRARY)

if(TENSORFLOW_FOUND)
  set(TENSORFLOW_LIBRARIES ${TENSORFLOW_LIBRARY})
  set(TENSORFLOW_INCLUDE_DIRS ${TENSORFLOW_INCLUDE_DIR})
endif()
