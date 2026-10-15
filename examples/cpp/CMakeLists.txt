cmake_minimum_required(VERSION 3.20)
project(dooms_chat LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(dooms 0.3 REQUIRED)   # -Ddooms_DIR=<sdk>/lib/cmake/dooms, or -DCMAKE_PREFIX_PATH=<sdk>
add_executable(chat chat.cpp)
target_link_libraries(chat PRIVATE dooms::dooms)
