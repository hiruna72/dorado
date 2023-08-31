if(APPLE)
    set(CPACK_GENERATOR "TGZ")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(CPACK_SYSTEM_NAME "osx-x64")
    else()
        set(CPACK_SYSTEM_NAME "osx-arm64")
    endif()
    set(CPACK_PRE_BUILD_SCRIPTS "${CMAKE_SOURCE_DIR}/cmake/DoradoCodesigning.cmake")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_GENERATOR "TGZ")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*")
        set(CPACK_SYSTEM_NAME "linux-arm64")
    else()
        set(CPACK_SYSTEM_NAME "linux-x64")
    endif()
elseif (WIN32)
    set(CPACK_GENERATOR "ZIP")
    set(CPACK_SYSTEM_NAME "win64")
else()
    message(FATAL_ERROR "Unexpected archive build platform: expected OSX, UNIX, or WIN32")
endif()

set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_SOURCE_DIR}/archive")
set(CPACK_PACKAGE_VENDOR "Oxford Nanopore Technologies PLC")
set(CPACK_PACKAGE_VERSION "${DORADO_VERSION_MAJOR}.${DORADO_VERSION_MINOR}.${DORADO_VERSION_REV}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENCE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

include(CPack)
