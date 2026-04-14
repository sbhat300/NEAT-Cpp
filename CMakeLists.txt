cmake_minimum_required(VERSION 3.10.0)
project(NEAT VERSION 2.5.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp ${PROJECT_SOURCE_DIR}/include/neat.cpp ${Engine_SOURCES})