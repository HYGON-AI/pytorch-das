# ---------------------------------------------------------------------------
# HML (Hygon Math Library) + HygonDNN integration switch hub.
#
# Single source of truth for the 4 user-facing knobs:
#   USE_HML=ON          -> HML master switch: selects BLAS=HBLAS and turns on
#                          every HML component (HLAPACK / HSPARSE / HFFT / HVML)
#   BLAS=HBLAS          -> legacy spelling of USE_HML=ON (still honoured)
#   USE_HDNN=ON         -> hygonDNN replaces bundled oneDNN (independent of HML)
#   HML_LINK_STATIC=ON  -> static link HML archives (HML only)
#
# Included from the top-level CMakeLists.txt after EnvVarForwarding (so a
# USE_HML / BLAS env var is already visible) and before cmake/Dependencies.cmake
# (so that the BLAS selection made here is what Dependencies.cmake sees).
# ---------------------------------------------------------------------------

# ---- HML master switch ----
option(USE_HML "Build with Hygon HML (HBLAS + HLAPACK/HSPARSE/HFFT/HVML) as the CPU math backend" OFF)
if(DEFINED ENV{USE_HML} AND "$ENV{USE_HML}" STREQUAL "ON")
  set(USE_HML ON CACHE BOOL "" FORCE)
endif()

# USE_HML=ON selects HBLAS; Dependencies.cmake must see BLAS=HBLAS from here.
if(USE_HML AND NOT BLAS STREQUAL "HBLAS")
  message(STATUS "USE_HML=ON: selecting BLAS=HBLAS (was '${BLAS}')")
  set(BLAS "HBLAS" CACHE STRING "BLAS library to use" FORCE)
endif()

if(USE_HML OR BLAS STREQUAL "HBLAS")
  set(USE_HML     ON  CACHE BOOL "" FORCE)   # BLAS=HBLAS implies USE_HML=ON
  set(USE_HLAPACK ON  CACHE BOOL "" FORCE)
  set(USE_HSPARSE ON  CACHE BOOL "" FORCE)
  set(USE_HFFT    ON  CACHE BOOL "" FORCE)
  set(USE_HVML    ON  CACHE BOOL "" FORCE)
  set(AT_HML_ENABLED 1)        # -> aten/src/ATen/Config.h; also gates has_spectral (HFFT is mandatory here)
else()
  set(USE_HML     OFF CACHE BOOL "" FORCE)
  set(USE_HLAPACK OFF CACHE BOOL "" FORCE)
  set(USE_HSPARSE OFF CACHE BOOL "" FORCE)
  set(USE_HFFT    OFF CACHE BOOL "" FORCE)
  set(USE_HVML    OFF CACHE BOOL "" FORCE)
  set(AT_HML_ENABLED 0)
endif()

# ---- STATIC (only meaningful with HML) ----
option(HML_LINK_STATIC "Link HML archives statically (.a)" OFF)
if(DEFINED ENV{HML_LINK_STATIC} AND "$ENV{HML_LINK_STATIC}" STREQUAL "ON")
  set(HML_LINK_STATIC ON CACHE BOOL "" FORCE)
endif()
if(HML_LINK_STATIC AND NOT USE_HML)
  message(WARNING "HML_LINK_STATIC=ON ignored: BLAS is not HBLAS")
  set(HML_LINK_STATIC OFF CACHE BOOL "" FORCE)
endif()

# ---- Component discovery (merged Find{HBLAS,HLAPACK,HSPARSE,HFFT,HVML}) ----
if(USE_HML)
  include(${CMAKE_CURRENT_LIST_DIR}/FindHML.cmake)
endif()

# ---- HygonDNN (independent of HML) ----
option(USE_HDNN "Use hygonDNN instead of the bundled oneDNN." OFF)
if(USE_HDNN)
  if(NOT LINUX OR NOT CPU_INTEL)
    message(FATAL_ERROR "USE_HDNN is only supported on Linux x86_64.")
  endif()
  set(USE_MKLDNN ON CACHE BOOL "Use MKLDNN" FORCE)
  add_compile_definitions(USE_HDNN)
endif()

# ---------------------------------------------------------------------------
# HML backend linking (moved out of cmake/Dependencies.cmake).
#
# Runs in the caller's scope (macro, not function) so that appends to
# Caffe2_DEPENDENCY_LIBS / Caffe2_PRIVATE_DEPENDENCY_LIBS and USE_LAPACK take
# effect. Assumes FindHML.cmake already populated <COMP>_LIB / <COMP>_INCLUDE_DIR.
# ---------------------------------------------------------------------------
macro(hml_configure_backends)
  # HLAPACK replaces the system LAPACK when HML is on.
  if(USE_HLAPACK)
    if(HLAPACK_FOUND)
      set(USE_LAPACK 1)
      include_directories(SYSTEM ${HLAPACK_INCLUDE_DIR})
      if(HML_LINK_STATIC)
        list(APPEND HML_STATIC_LIBS ${HLAPACK_LIB})
      else()
        list(APPEND Caffe2_DEPENDENCY_LIBS ${HLAPACK_LIB})
      endif()
    else()
      message(FATAL_ERROR "Could not find HLAPACK (USE_HLAPACK=ON)")
    endif()
  else()
    find_package(LAPACK)
    if(LAPACK_FOUND)
      set(USE_LAPACK 1)
      list(APPEND Caffe2_PRIVATE_DEPENDENCY_LIBS ${LAPACK_LIBRARIES})
    endif()
  endif()

  if(USE_HSPARSE)
    if(HSPARSE_FOUND)
      include_directories(SYSTEM ${HSPARSE_INCLUDE_DIR})
      if(HML_LINK_STATIC)
        list(APPEND HML_STATIC_LIBS ${HSPARSE_LIB})
      else()
        list(APPEND Caffe2_DEPENDENCY_LIBS ${HSPARSE_LIB})
        get_filename_component(HSPARSE_LIB_DIR ${HSPARSE_LIB} DIRECTORY)
        add_link_options(-Wl,-rpath-link,${HSPARSE_LIB_DIR} -Wl,-rpath,${HSPARSE_LIB_DIR})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${HSPARSE_LIB_DIR}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${HSPARSE_LIB_DIR}")
      endif()
    else()
      message(FATAL_ERROR "Could not find HSPARSE (USE_HSPARSE=ON)")
    endif()
  endif()

  if(USE_HFFT)
    if(HFFT_FOUND)
      include_directories(SYSTEM ${HFFT_INCLUDE_DIR})
      if(HML_LINK_STATIC)
        list(APPEND HML_STATIC_LIBS ${HFFT_LIB})
      else()
        list(APPEND Caffe2_DEPENDENCY_LIBS ${HFFT_LIB})
      endif()
    else()
      message(FATAL_ERROR "Could not find HFFT (USE_HFFT=ON)")
    endif()
  endif()

  if(USE_HVML)
    if(HVML_FOUND)
      include_directories(SYSTEM ${HVML_INCLUDE_DIR})
      # Hygon vML is called directly (vd*/vs* interface in ATen/cpu/hml/vml_hml.h);
      # no -mveclibabi compiler option needed. PUBLIC dep: the stubs live inside
      # libtorch_cpu and reference vd*/vs*.
      if(HML_LINK_STATIC)
        list(APPEND HML_STATIC_LIBS ${HVML_LIB})
      else()
        list(APPEND Caffe2_DEPENDENCY_LIBS ${HVML_LIB})
        get_filename_component(HVML_LIB_DIR ${HVML_LIB} DIRECTORY)
        add_link_options(-L${HVML_LIB_DIR} -Wl,-rpath-link,${HVML_LIB_DIR} -Wl,-rpath,${HVML_LIB_DIR} -fno-lto)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath-link,${HVML_LIB_DIR}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath-link,${HVML_LIB_DIR}")
      endif()
    else()
      message(FATAL_ERROR "Could not find HVML (USE_HVML=ON)")
    endif()
  endif()

  if(HML_LINK_STATIC AND HML_STATIC_LIBS)
    # Static link: all HML archives in one group (hblas<->hlm mutual refs),
    # plus system deps. libgomp stays dynamic (OpenMP runtime).
    get_filename_component(HML_STATIC_DIR ${HBLAS_LIB} DIRECTORY)
    list(APPEND HML_STATIC_LIBS ${HML_STATIC_DIR}/libhlm.a)
    list(APPEND Caffe2_DEPENDENCY_LIBS
         -Wl,--start-group ${HML_STATIC_LIBS} -Wl,--end-group
         -lgfortran -lgomp -lquadmath -lm -lpthread)
    message(STATUS "HML_LINK_STATIC=ON: ${HML_STATIC_LIBS}")
  endif()
endmacro()
