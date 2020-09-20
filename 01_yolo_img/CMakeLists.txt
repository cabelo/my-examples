cmake_minimum_required(VERSION 3.10.2)
project(yoloimg)
find_package(OpenCV REQUIRED)

set(TARGET yoloimg)
set(SOURCES src/yoloimg.cpp)

add_executable(${TARGET} ${SOURCES})

get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
  # only make run target available for stand-alone build
  add_custom_target(run ${TARGET} ${RUNARGS})
endif()

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(${TARGET}  ${OpenCV_LIBS})

