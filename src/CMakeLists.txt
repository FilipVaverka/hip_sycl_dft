## ======== Copyright (c) 2022, Filip Vaverka, All rights reserved. ======== ##
##
## Purpose:     Common library interface
##
## $NoKeywords: $HipSyclDft $CMakeLists.txt
## $Date:       $2022-08-08
## ========================================================================= ##

set(INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

set(SUB_SRCS
    sycl_dft.cpp
)

PREPEND(SUB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${SUB_SRCS})
set(SRCS_LIST ${SRCS_LIST} ${SUB_SRCS} PARENT_SCOPE)