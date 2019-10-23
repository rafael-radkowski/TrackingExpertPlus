

find_package(Eigen3 REQUIRED)

# eigen
include_directories(${EIGEN3_INCLUDE_DIR})

# local 
include_directories(${CMAKE_SOURCE_DIR}/include/camera)
include_directories(${CMAKE_SOURCE_DIR}/include/detection)
include_directories(${CMAKE_SOURCE_DIR}/include/kdtree)
include_directories(${CMAKE_SOURCE_DIR}/include/loader)
include_directories(${CMAKE_SOURCE_DIR}/include/nearest_neighbors)
include_directories(${CMAKE_SOURCE_DIR}/include/pointcloud)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)


add_subdirectory(gl_support)