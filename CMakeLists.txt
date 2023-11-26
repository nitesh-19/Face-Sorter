# CMakeList.txt : CMake project for CMakeProject1, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


# Disable Optimization


# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Media-Organizer")

# Add source to this project's executable.
add_executable (Media-Organizer "Media-Organizer.cpp")
target_include_directories( Media-Organizer PRIVATE "C:/Program Files (x86)/dlib_project/include")
target_link_directories(Media-Organizer PRIVATE "C:/Program Files (x86)/dlib_project/lib")
target_link_libraries(Media-Organizer PRIVATE dlib19.24.0_release_64bit_msvc1936)
if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Media-Organizer PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
