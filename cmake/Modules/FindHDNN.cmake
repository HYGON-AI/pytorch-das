# - Find HDNN (HygonDNN library, oneDNN-compatible)
# HygonDNN is Hygon's oneDNN-compatible deep neural network library, used in
# place of the bundled oneDNN.
#
# This module sets:
#   HDNN_FOUND       - TRUE if found
#   HDNN_LIBRARY     - path to libdnnl.so
#   HDNN_INCLUDE_DIR - path to dnnl.h

SET(HDNN_INCLUDE_SEARCH_PATHS
  $ENV{HDNN_HOME}/include
  /usr/local/include
)

SET(HDNN_LIB_SEARCH_PATHS
  $ENV{HDNN_HOME}/lib64
  $ENV{HDNN_HOME}/lib
  /usr/local/lib
  /usr/local/lib64
  /usr/lib
  /usr/lib64
)

FIND_PATH(HDNN_INCLUDE_DIR NAMES dnnl.h PATHS ${HDNN_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(HDNN_LIBRARY NAMES dnnl PATHS ${HDNN_LIB_SEARCH_PATHS})

SET(HDNN_FOUND ON)

IF(NOT HDNN_INCLUDE_DIR)
  SET(HDNN_FOUND OFF)
  MESSAGE(STATUS "Could not find HDNN include (dnnl.h). Turning HDNN_FOUND off")
ENDIF()

IF(NOT HDNN_LIBRARY)
  SET(HDNN_FOUND OFF)
  MESSAGE(STATUS "Could not find HDNN lib (libdnnl). Turning HDNN_FOUND off")
ENDIF()

IF(HDNN_FOUND)
  IF(NOT HDNN_FIND_QUIETLY)
    MESSAGE(STATUS "Found HDNN library: ${HDNN_LIBRARY}")
    MESSAGE(STATUS "Found HDNN include: ${HDNN_INCLUDE_DIR}")
  ENDIF()
ELSE()
  IF(HDNN_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find HDNN")
  ENDIF()
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(HDNN DEFAULT_MSG HDNN_INCLUDE_DIR HDNN_LIBRARY)

MARK_AS_ADVANCED(HDNN_INCLUDE_DIR HDNN_LIBRARY)

find_path(IDEEP_INCLUDE_DIR ideep.hpp
          PATHS "${PROJECT_SOURCE_DIR}/third_party/ideep"
          PATH_SUFFIXES include
          NO_DEFAULT_PATH)
if(NOT IDEEP_INCLUDE_DIR)
  message(FATAL_ERROR "ideep headers are required to use hygonDNN.")
endif()

set(MKLDNN_INCLUDE_DIR "${HDNN_INCLUDE_DIR};${IDEEP_INCLUDE_DIR}")
set(MKLDNN_LIBRARIES "${HDNN_LIBRARY}")
set(MKLDNN_FOUND TRUE)
set(BUILD_ONEDNN_GRAPH ON)
set(AT_MKLDNN_ACL_ENABLED 0)

if(NOT TARGET caffe2::mkldnn)
  add_library(caffe2::mkldnn INTERFACE IMPORTED)
endif()
set_property(TARGET caffe2::mkldnn PROPERTY INTERFACE_INCLUDE_DIRECTORIES
             "${MKLDNN_INCLUDE_DIR}")
set_property(TARGET caffe2::mkldnn PROPERTY INTERFACE_LINK_LIBRARIES
             "${MKLDNN_LIBRARIES}")
if(TARGET caffe2::openmp)
  set_property(TARGET caffe2::mkldnn APPEND PROPERTY INTERFACE_LINK_LIBRARIES
               caffe2::openmp)
endif()

# Install libdnnl.so / .so.3 / .so.3.8 from the HDNN install tree.
GET_FILENAME_COMPONENT(HDNN_LIB_DIR ${HDNN_LIBRARY} DIRECTORY)
install(FILES
        "${HDNN_LIB_DIR}/libdnnl.so.3.8"
        "${HDNN_LIB_DIR}/libdnnl.so.3"
        "${HDNN_LIB_DIR}/libdnnl.so"
        DESTINATION lib)

MESSAGE(STATUS "Found hygonDNN: ${HDNN_LIBRARY}")
