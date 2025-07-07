cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(aimbot LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math -maxrregcount=64")

# Enable link-time optimization for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(CUDA REQUIRED)

# TensorRT paths (adjust as needed)
set(TENSORRT_ROOT "C:/TensorRT" CACHE PATH "TensorRT root directory")
find_library(TENSORRT_LIBRARY nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(TENSORRT_PARSER_LIBRARY nvonnxparser HINTS ${TENSORRT_ROOT}/lib)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/Aimbot.cpp
    src/Aiming.cpp
    src/ConfigManager.cpp
    src/InferenceEngine.cpp
    src/Logger.cpp
    src/Path.cpp
    src/PerformanceMonitor.cpp
    src/ScreenCapture.cpp
    src/UI.cpp
    src/GPUBufferPool.cpp
    ghub/ghub.cpp
)

# CUDA sources
set(CUDA_SOURCES
    src/CudaKernels.cu
    src/OptimizedCudaKernels.cu
)

# Create separate libraries for better build organization
add_library(cuda_kernels STATIC ${CUDA_SOURCES})
target_compile_features(cuda_kernels PUBLIC cxx_std_17)

add_library(inference_engine STATIC 
    src/InferenceEngine.cpp
    src/GPUBufferPool.cpp
)
target_link_libraries(inference_engine cuda_kernels ${TENSORRT_LIBRARY} ${TENSORRT_PARSER_LIBRARY})

add_library(ui_components STATIC 
    src/UI.cpp
    src/PerformanceMonitor.cpp
)

# Main executable
add_executable(aimbot ${SOURCES})

# Link libraries
target_link_libraries(aimbot
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PARSER_LIBRARY}
    cuda_kernels
    inference_engine
    ui_components
    nvml
    d3d11
    dxgi
    user32
    gdi32
)

# CUDA specific settings
set_property(TARGET aimbot PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_kernels PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Windows specific settings
if(WIN32)
    target_compile_definitions(aimbot PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    set_property(TARGET aimbot PROPERTY WIN32_EXECUTABLE TRUE)
endif()

# Copy DLLs to output directory for Windows
if(WIN32)
    # Copy TensorRT DLLs
    file(GLOB TENSORRT_DLLS "${TENSORRT_ROOT}/lib/*.dll")
    foreach(dll ${TENSORRT_DLLS})
        add_custom_command(TARGET aimbot POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${dll} $<TARGET_FILE_DIR:aimbot>)
    endforeach()
    
    # Copy CUDA DLLs
    find_file(CUDART_DLL cudart64_*.dll HINTS ${CUDA_TOOLKIT_ROOT_DIR}/bin)
    if(CUDART_DLL)
        add_custom_command(TARGET aimbot POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CUDART_DLL} $<TARGET_FILE_DIR:aimbot>)
    endif()
endif()

# Install targets
install(TARGETS aimbot DESTINATION bin)
install(DIRECTORY models/ DESTINATION bin/models OPTIONAL)
install(DIRECTORY fonts/ DESTINATION bin/fonts OPTIONAL)