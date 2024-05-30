# Author: Nicola Maritan

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(sport_video_analysis_for_biliard_matches
    VERSION
        1.0
    DESCRIPTION
        ""
    LANGUAGES
        CXX
)

find_package(OpenCV REQUIRED)
include_directories(include ${OpenCV_INCLUDE_DIRS}) 

add_library(show_cat
    include/show_cat.h
    src/show_cat.cpp
)

add_executable(${PROJECT_NAME}
	src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    show_cat
    ${OpenCV_LIBS}
)