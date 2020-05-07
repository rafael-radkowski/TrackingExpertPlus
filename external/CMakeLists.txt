
# Main cmake file 
cmake_minimum_required(VERSION 2.6)

# policy for relative subfolders
cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0010 NEW)


#--------------------------------------------------------------
# Build GLFW
# Options
if (TRAKINGX_BUILD_GLFW)
    add_subdirectory(glfw)
endif()
#--------------------------------------------------------------


#--------------------------------------------------------------
# Build GLEW
# Options
if (TRAKINGX_BUILD_GLEW)
    add_subdirectory(glew)
endif()
#--------------------------------------------------------------


# enable visual studio folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set( OutputDir ${CMAKE_SOURCE_DIR}/lib)


set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(OUTPUT_BINDIR ${CMAKE_SOURCE_DIR}/bin)
make_directory(${OUTPUT_BINDIR})

# Set the project architecture for this example
set( PROJECT_ARCH "x64" )

# All output files are copied to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)



# Set the runtime output path
set( RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")

# All library output files are copied to dep/lib
#set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)
set(LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/lib)


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find packages
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(glew REQUIRED)
FIND_PACKAGE(glm REQUIRED)
FIND_PACKAGE(GLFW3 REQUIRED)
find_package(Eigen3 REQUIRED)

# eigen
include_directories(${EIGEN3_INCLUDE_DIR})

# local 
include_directories(${CMAKE_SOURCE_DIR}/include/camera)
include_directories(${CMAKE_SOURCE_DIR}/include/detection)
include_directories(${CMAKE_SOURCE_DIR}/include/kdtree)
include_directories(${CMAKE_SOURCE_DIR}/include/nearest_neighbors)
include_directories(${CMAKE_SOURCE_DIR}/include/pointcloud)
include_directories(${CMAKE_SOURCE_DIR}/include/utils)
include_directories(${CMAKE_SOURCE_DIR}/include/loader)


include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLFW3_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIR})
include_directories(gl_ext)
include_directories(gl_support/include)




file(GLOB GLUtils_SRC
    "gl_support/src/*.cpp"
	"gl_ext/*.cpp"
)

file(GLOB GLUtils_INC
    "gl_support/include/*.h"
    "gl_ext/*.h"
)


source_group(src FILES ${GLUtils_SRC} )
source_group(include FILES ${GLUtils_INC} )


add_definitions(-DWITH_EXPORT)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OutputDir})
set(CMAKE_DEBUG_POSTFIX d)



add_compile_definitions(_WITH_CAMERA)
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_definitions(-DWITH_EXPORT)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OutputDir})
set(CMAKE_DEBUG_POSTFIX d)


# Create an executable
add_library(GLUtils  ${GLUtils_SRC} ${GLUtils_INC} )

# glfw dependency
if (BUILD_GLFW)
	add_dependencies(GLUtils glfw)
endif()

# glew dependency
if (BUILD_GLEW)
    add_dependencies(GLUtils glew)
endif()

# Add libraries
target_link_libraries(GLUtils   ${GLEW_LIBS} ${GLEW_LIBS} ${GLFW3_LIBS} ${OPENGL_LIBRARIES}   )


set_target_properties( GLUtils PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OutputDir} )
set_target_properties( GLUtils PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OutputDir} )

set_target_properties (GLUtils PROPERTIES
    FOLDER External
)


#################################################################################################



# Add all files to the configuration
file(GLOB Test_SRC
	gl_support/test/gl_test_light/main.cpp
)


# Add link directories
# Note required for this project
link_directories(${CMAKE_SOURCE_DIR}/lib)


set(TEST_APP0 GLUtils_Test)

# Create an executable
add_executable(${TEST_APP0} ${Test_SRC} )


# glfw dependency
if (BUILD_GLFW)
	add_dependencies(${TEST_APP0} glfw)
endif()

# glew dependency
if (BUILD_GLEW)
    add_dependencies(${TEST_APP0} glew)
endif()

add_dependencies(${TEST_APP0} GLUtils)

# Add libraries
target_link_libraries(${TEST_APP0}  ${GLEW_LIBS} ${GLEW_LIBS} ${GLFW3_LIBS} ${OPENGL_LIBS} ${OPENGL_LIBRARIES}  )
target_link_libraries(${TEST_APP0} debug  GLUtilsd.lib )
target_link_libraries(${TEST_APP0} optimized  GLUtils.lib )


set_target_properties (GLUtils_Test PROPERTIES
    FOLDER External
)




