# Main cmake file 


cmake_minimum_required(VERSION 2.6)

cmake_policy(SET CMP0015 NEW)
cmake_policy(SET CMP0074 NEW)


# Find packages
FIND_PACKAGE(OpenGL REQUIRED)



set(GLEW_SRC
    glew.c
	../glew.rc
)

set(GLEW_INC
    ../include/GL/glew.h
	../include/GL/wglew.h
)


source_group(src FILES ${GLEW_SRC} )
source_group(include FILES ${GLEW_INC} )


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/bin")
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/bin")


set(CMAKE_DEBUG_POSTFIX d)

#message(${CMAKE_CURRENT_SOURCE_DIR})
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../include)


# Create an executable
add_library(${PROJECT_NAME} SHARED ${GLEW_INC} ${GLEW_SRC} )


# Add libraries
target_link_libraries(${PROJECT_NAME}   ${OPENGL_LIBRARIES}   )


set( OutputDir ${CMAKE_CURRENT_SOURCE_DIR}/../lib)
set_target_properties( ${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${OutputDir} )
set_target_properties( ${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${OutputDir} )

set_target_properties (${PROJECT_NAME} PROPERTIES
    FOLDER External
)