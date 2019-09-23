# This script finds cub
#Rafael Radkowski


set(CUB_INSTALL_PATHS_  "C:/SDK/cub-1.8.0"
						"C:/SDK/cub-1.6.4" 
						"C:/SDK/cub"  
						"D:/SDK/cub-1.7.0"
						"D:/SDK/cub-1.6.4"
						)


find_path (CUB_INCLUDE_DIR NAMES cub/cub.cuh
		PATHS ${CUB_INSTALL_PATHS_} )


if (NOT CUB_INCLUDE_DIR) 
	set (CUB_FOUND false)
	 message("ERROR: CUB NOT found!")
endif(NOT CUB_INCLUDE_DIR) 
if ( CUB_INCLUDE_DIR) 
	set (CUB_FOUND true)
	 message("Found CUB in ${CUB_INCLUDE_DIR}.")
endif( CUB_INCLUDE_DIR) 