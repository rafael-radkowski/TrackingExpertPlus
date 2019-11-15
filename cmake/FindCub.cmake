###############################################################################
# Find CUB (CUDA Unbound)
#
# Exports the target CUB::CUB
# Recommended method to link is using target_link_libraries(... CUB::CUB).
# CUB is a header-only library, so it will just add to your include directories
# No extra steps are required if using this method.
# 
# Alternatively, can also use old-style by making use of the following defined variables:
# CUB_FOUND - True if CUB was found.
# CUB_INCLUDE_DIRS - Directories containing the CUB include files.

# In CMake 3.12+, set the CUB_ROOT environment variable to add another search path here
find_path(
    CUB_ROOT_DIR
    NAMES cub/cub.cuh
    PATHS "C:/SDK/cub-1.8.0"
)

mark_as_advanced(
    CUB_FOUND
    CUB_INCLUDE_DIRS
)

#include(FindPackageHandleStandardArgs)
#find_package_handle_standard_args(
 #   CUB
 #   DEFAULT_MSG
 #   CUB_ROOT_DIR
#)

if(CUB_FOUND)
    set(CUB_INCLUDE_DIRS ${CUB_ROOT_DIR})
endif()

if(CUB_FOUND AND NOT TARGET CUB::CUB)
    add_library(CUB::CUB INTERFACE IMPORTED)
    set_target_properties(CUB::CUB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CUB_INCLUDE_DIRS}"
    )
endif()