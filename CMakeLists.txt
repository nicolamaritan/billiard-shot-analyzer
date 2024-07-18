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

add_library(playing_field_localization
    include/playing_field_localization.h
    src/playing_field_localization.cpp
)

add_library(balls_localization
    include/balls_localization.h
    src/balls_localization.cpp
)

add_library(geometry
    include/geometry.h
    src/geometry.cpp
)

add_library(segmentation
    include/segmentation.h
    src/segmentation.cpp
)

add_library(minimap
    include/minimap.h
    src/minimap.cpp
)

add_library(frame_segmentation
    include/frame_segmentation.h
    src/frame_segmentation.cpp
)

add_library(frame_detection
    include/frame_detection.h
    src/frame_detection.cpp
)

add_library(performance_measurement
    include/performance_measurement.h
    src/performance_measurement.cpp
)

add_library(dataset_evaluation
    include/dataset_evaluation.h
    src/dataset_evaluation.cpp
)

add_executable(${PROJECT_NAME}
	src/main.cpp
)

add_executable(generate_performance
	src/generate_performance.cpp
)

add_executable(generate_masks_and_detection
	src/generate_masks_and_detection.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    show_cat
    playing_field_localization
    balls_localization
    geometry
    segmentation
    minimap
    dataset_evaluation
)

target_link_libraries(generate_performance
    ${OpenCV_LIBS}
    show_cat
    dataset_evaluation
    performance_measurement
    frame_segmentation
    frame_detection
    playing_field_localization
    balls_localization
    geometry
    segmentation
)

target_link_libraries(generate_masks_and_detection
    ${OpenCV_LIBS}
    show_cat
    dataset_evaluation
    performance_measurement
    frame_segmentation
    frame_detection
    playing_field_localization
    balls_localization
    geometry
    segmentation
)