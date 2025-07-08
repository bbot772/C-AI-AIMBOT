cmake_minimum_required(VERSION 3.18)

project(Aimbot LANGUAGES CXX CUDA)

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# MSVC-specific settings
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /EHsc")
endif()

# Find necessary packages
find_package(OpenCV REQUIRED HINTS "C:/Users/Fantasir/Desktop/New C++ AImbot/opencv/build")

# Manually configure CUDA & TensorRT paths since find_package can be unreliable
set(CUDA_TOOLKIT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(TensorRT_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.12.0.36")

if(NOT EXISTS "${CUDA_TOOLKIT_ROOT}/include")
    message(FATAL_ERROR "CUDA include directory not found: ${CUDA_TOOLKIT_ROOT}/include")
endif()
if(NOT EXISTS "${TensorRT_ROOT}/include")
    message(FATAL_ERROR "TensorRT include directory not found: ${TensorRT_ROOT}/include")
endif()

# List all source files for the project
set(SOURCES
    src/Aimbot.cpp
    src/Aiming.cpp
    src/ConfigManager.cpp
    src/Logger.cpp
    src/main.cpp
    src/Path.cpp
    src/PerformanceMonitor.cpp
    src/ScreenCapture.cpp
    src/InferenceEngine.cpp
    src/UI.cpp
    src/CudaKernels.cu
    ghub/ghub.cpp
    imgui-master/imgui.cpp
    imgui-master/imgui_draw.cpp
    imgui-master/imgui_tables.cpp
    imgui-master/imgui_widgets.cpp
    imgui-master/backends/imgui_impl_dx11.cpp
    imgui-master/backends/imgui_impl_win32.cpp
    include/implot.cpp
    include/implot_items.cpp
)

# Create the executable target
add_executable(Aimbot ${SOURCES})

# Specify include directories for the target
target_include_directories(Aimbot PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/ghub"
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui-master"
    "${CMAKE_CURRENT_SOURCE_DIR}/imgui-master/backends"
    "${TensorRT_ROOT}/include"
    "${CUDA_TOOLKIT_ROOT}/include"
    ${OpenCV_INCLUDE_DIRS}
)

# Link all necessary libraries to the target
target_link_libraries(Aimbot PRIVATE
    ${OpenCV_LIBS}
    "${TensorRT_ROOT}/lib/nvinfer_10.lib"
    "${TensorRT_ROOT}/lib/nvinfer_plugin_10.lib"
    "${TensorRT_ROOT}/lib/nvonnxparser_10.lib"
    "${CUDA_TOOLKIT_ROOT}/lib/x64/cudart.lib"
    "${CUDA_TOOLKIT_ROOT}/lib/x64/nvml.lib"
    d3d11
    dxgi
    dwmapi
)

# Set properties for Windows executable
if(WIN32)
    set_target_properties(Aimbot PROPERTIES WIN32_EXECUTABLE ON)
endif()

# Required definitions for ImGui backends
target_compile_definitions(Aimbot PRIVATE
    IMGUI_IMPL_WIN32_DISABLE_GAMEPAD
    IMGUI_IMPL_WIN32_DISABLE_LINK_TIME_FUNCTIONS
)

# Set the output directory for the final executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release)
