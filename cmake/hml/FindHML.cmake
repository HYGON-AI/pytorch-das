# ---------------------------------------------------------------------------
# FindHML.cmake - merged finder for Hygon Math Library (HML) components.
#
# Replaces the five near-identical modules
#   FindHBLAS / FindHLAPACK / FindHSPARSE / FindHFFT / FindHVML .cmake
#
# For each component <C> it sets:
#   <C>_FOUND, <C>_LIB, <C>_INCLUDE_DIR
# and an aggregate HML_FOUND (all components found).
#
# Honors HML_LINK_STATIC (or $ENV{HML_LINK_STATIC}=ON) to look up .a archives.
# Search roots are $ENV{<C>_HOME}/{include,lib} plus the usual system paths.
# ---------------------------------------------------------------------------

macro(hml_find_component NAME HEADER ENV_HOME)
  find_path(${NAME}_INCLUDE_DIR NAMES ${HEADER}
    PATHS $ENV{${ENV_HOME}}/include /usr/local/include)

  set(_hml_lib_paths
    $ENV{${ENV_HOME}}/lib
    /usr/local/lib /usr/local/lib64 /usr/lib /usr/lib64)

  if(HML_LINK_STATIC OR "$ENV{HML_LINK_STATIC}" STREQUAL "ON")
    set(_hml_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(${NAME}_LIB NAMES ${ARGN} PATHS ${_hml_lib_paths})
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_hml_saved_suffixes}")
  else()
    find_library(${NAME}_LIB NAMES ${ARGN} PATHS ${_hml_lib_paths})
  endif()

  set(${NAME}_FOUND ON)
  if(NOT ${NAME}_INCLUDE_DIR)
    set(${NAME}_FOUND OFF)
    message(STATUS "Could not find ${NAME} include (${HEADER}). Turning ${NAME}_FOUND off")
  endif()
  if(NOT ${NAME}_LIB)
    set(${NAME}_FOUND OFF)
    message(STATUS "Could not find ${NAME} lib. Turning ${NAME}_FOUND off")
  endif()
  if(${NAME}_FOUND)
    message(STATUS "Found ${NAME} library: ${${NAME}_LIB}")
    message(STATUS "Found ${NAME} include: ${${NAME}_INCLUDE_DIR}")
  endif()
  mark_as_advanced(${NAME}_INCLUDE_DIR ${NAME}_LIB)
endmacro()

hml_find_component(HBLAS   cblas.h      HBLAS_HOME   hblas)
hml_find_component(HLAPACK lapacke.h    HLAPACK_HOME hlapack_lp64)
hml_find_component(HSPARSE hml_sparse.h HSPARSE_HOME hsparse)
hml_find_component(HFFT    fftw3.h      HFFT_HOME    fftw3)
hml_find_component(HVML    hml_vml.h    HVML_HOME    hvml vml-hg)

set(HML_FOUND ON)
foreach(_hml_component HBLAS HLAPACK HSPARSE HFFT HVML)
  if(NOT ${_hml_component}_FOUND)
    set(HML_FOUND OFF)
  endif()
endforeach()
unset(_hml_component)
