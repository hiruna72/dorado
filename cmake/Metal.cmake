download_and_extract(https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13_iOS16.zip metal-cpp)
find_library(APPLE_FWK_FOUNDATION Foundation REQUIRED)
find_library(APPLE_FWK_QUARTZ_CORE QuartzCore REQUIRED)
find_library(APPLE_FWK_METAL Metal REQUIRED)
find_library(IOKIT IOKit REQUIRED)

set(AIR_FILES)
set(METAL_SOURCES dorado/nn/metal/nn.metal)

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(XCRUN_SDK iphoneos)
else()
    set(XCRUN_SDK macosx)
endif()

foreach(source ${METAL_SOURCES})
    get_filename_component(basename "${source}" NAME_WE)
    set(air_path "${CMAKE_BINARY_DIR}/${basename}.air")
    add_custom_command(
        OUTPUT "${air_path}"
        COMMAND xcrun -sdk ${XCRUN_SDK} metal -ffast-math -c "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${air_path}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
        COMMENT "Compiling metal kernels"
    )
    list(APPEND AIR_FILES "${air_path}")
endforeach()

add_custom_command(
    OUTPUT default.metallib
    COMMAND xcrun -sdk ${XCRUN_SDK} metallib ${AIR_FILES} -o ${CMAKE_BINARY_DIR}/lib/default.metallib
    DEPENDS ${AIR_FILES}
    COMMENT "Creating metallib"
)
install(FILES ${CMAKE_BINARY_DIR}/lib/default.metallib DESTINATION lib)
