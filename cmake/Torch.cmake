set(TORCH_VERSION 1.13.1)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR WIN32)
    find_package(CUDAToolkit)
    # the torch cuda.cmake will set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}") [2]
    # so we need to make CUDA_TOOLKIT_ROOT_DIR is set correctly as per [1] we also
    # 1. https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
    # 2. https://github.com/pytorch/pytorch/blob/5fa71207222620b4efb78989849525d4ee6032e8/cmake/public/cuda.cmake#L40
    set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
      set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
    endif()

    set(CUDNN_LIBRARY_PATH ${DORADO_3RD_PARTY}/fake_cudnn)
    set(CUDNN_INCLUDE_PATH ${DORADO_3RD_PARTY}/fake_cudnn)
    set(CMAKE_CUDA_ARCHITECTURES 70 72 75 80 86)

    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.4)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 87)
    endif()
    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.8)
      list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
    endif()
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(DORADO_USING_OLD_CPP_ABI)
        set(TORCH_URL https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-${TORCH_VERSION}%2Bcu117.zip)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-no-cxx11-abi-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
    else()
        set(TORCH_URL https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-${TORCH_VERSION}%2Bcu117.zip)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
    endif()
elseif(APPLE)
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
        set(TORCH_URL https://nanoporetech.box.com/shared/static/nzdq2wk45pzbwi2zex92j28dt3s5k9vt.tgz)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}")
    else()
        set(TORCH_URL https://files.pythonhosted.org/packages/2d/ab/8210e877debc6e16c5f64345b08abfd667ade733329ef8b38dd06a362513/torch-${TORCH_VERSION}-cp39-none-macosx_11_0_arm64.whl)
        set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/torch")
    endif()
elseif(WIN32)
    set(TORCH_URL https://download.pytorch.org/libtorch/cu117/libtorch-win-shared-with-deps-${TORCH_VERSION}%2Bcu117.zip)
    set(TORCH_LIB "${DORADO_3RD_PARTY}/torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME}/libtorch")
    add_compile_options(
        # Note we need to use the generator expression to avoid setting this for CUDA.
        $<$<COMPILE_LANGUAGE:CXX>:/wd4624> # from libtorch: destructor was implicitly defined as deleted 
    )
endif()

if(DEFINED DORADO_LIBTORCH_DIR)
    # Use the existing libtorch we have been pointed at
    list(APPEND CMAKE_PREFIX_PATH ${DORADO_LIBTORCH_DIR})
    message(STATUS "Using existing libtorch at ${DORADO_LIBTORCH_DIR}")
    set(TORCH_LIB ${DORADO_LIBTORCH_DIR})
else()
    # Get libtorch (if we don't already have it)
    if(DORADO_USING_OLD_CPP_ABI AND NOT WIN32 AND NOT APPLE)
        download_and_extract(${TORCH_URL} torch-no-cxx11-abi-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME})
    else()
        download_and_extract(${TORCH_URL} torch-${TORCH_VERSION}-${CMAKE_SYSTEM_NAME})
    endif()
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_LIB}")
endif()

find_package(Torch REQUIRED)

if(WIN32 AND DEFINED MKL_ROOT)
    link_directories(${MKL_ROOT}/lib/intel64)
endif()
