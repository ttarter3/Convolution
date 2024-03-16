# CPE613_assignment2

cmake_minimum_required(VERSION 3.10)

SET(ProjectId "CPE613_assignment2_script")
project(${ProjectId})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add ExternalProject module
#include(ExternalProject)

# Set the URL for your external dependency
#set(EXTERNAL_PROJECT_URL "https://github.com/opencv/opencv.git")

# Set the destination directory for the external project
#set(EXTERNAL_PROJECT_PREFIX "${CMAKE_SOURCE_DIR}/partythird/opencv")

# ExternalProject_Add command
#ExternalProject_Add(
#    opencv_project
#    PREFIX ${EXTERNAL_PROJECT_PREFIX}
#    GIT_REPOSITORY ${EXTERNAL_PROJECT_URL}
#    CMAKE_ARGS
#      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_SOURCE_DIR}/install/opencv
    # You can add more options as needed, such as GIT_TAG for a specific version
    # CONFIGURE_COMMAND ""
    # BUILD_COMMAND ""
    # INSTALL_COMMAND ""
    # TEST_COMMAND ""
#)

# Add include directories and link against the external project
#include_directories(${EXTERNAL_PROJECT_PREFIX}/include)
#link_directories(${EXTERNAL_PROJECT_PREFIX}/lib)

# Set header files (C++ files)
file(GLOB_RECURSE _HDRS "include/*.h*")
file(GLOB_RECURSE _HDRS_cuda "include/*.cuh")

# Set source files (C++ files)
file(GLOB_RECURSE CPP_SOURCES "src/*.cpp")

# Set CUDA source files
file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

# Set include directories
include_directories(include)

# Find CUDA package
find_package(CUDA REQUIRED)
find_package(cblas)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

list( APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib )
#set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories("/data001/heterogene_mw/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/cblas-2015-06-06-xm3olrbm2m62idlfxvte3vg55fm766ma/include")
include_directories("/data001/heterogene_mw/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/openblas-0.3.26-awiuvkvry65dlq4hgj6oudf6ufo2iffn/include")

link_directories("/data001/heterogene_mw/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/cblas-2015-06-06-xm3olrbm2m62idlfxvte3vg55fm766ma/lib")
link_directories("/data001/heterogene_mw/spack/opt/spack/linux-rocky8-zen/gcc-8.5.0/openblas-0.3.26-awiuvkvry65dlq4hgj6oudf6ufo2iffn/lib")

file( GLOB APP_SOURCES "./*.cpp" )
message("====================================================================}")
message("APP_SOURCES: ${APP_SOURCES}")
message("====================================================================}")
foreach( testsourcefile ${APP_SOURCES} )
    # Cut off the file extension and directory path
    get_filename_component( testname ${testsourcefile} NAME_WE )
    add_executable( ${testname} ${testsourcefile} )
    
    # Make sure YourLib is linked to each app
    # You can link against other libraries if needed
    target_link_libraries(${testname} cudasupportfunctions)
    target_link_libraries(${testname} Convolution)
    target_link_libraries(${testname} cblas)
    target_link_libraries(${testname} openblas)
    target_link_libraries(${testname} CUDA::cudart)
    target_link_libraries(${testname} CUDA::cublas)
    target_link_libraries(${testname} OpenMP::OpenMP_CXX) # Link OpenMP library


    link_directories(${CMAKE_SOURCE_DIR}/install/lib)
    link_directories(${CMAKE_SOURCE_DIR}/build)
    
    # Install the ProjectA executable
    install(TARGETS ${testname} DESTINATION bin)
    message("testname: ${testname}")
    
    
    # Specify that YourExecutable depends on your_dependency_project
    #add_dependencies(${testname} opencv_project)
endforeach( testsourcefile ${APP_SOURCES} )



