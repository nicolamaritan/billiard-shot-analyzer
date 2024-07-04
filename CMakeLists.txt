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

add_library(playing_field_localizer
    include/playing_field_localizer.h
    src/playing_field_localizer.cpp
)

add_library(balls_localizer
    include/balls_localizer.h
    src/balls_localizer.cpp
)

add_library(geometry
    include/geometry.h
    src/geometry.cpp
)


add_executable(${PROJECT_NAME}
	src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    show_cat
    playing_field_localizer
    balls_localizer
    geometry
)