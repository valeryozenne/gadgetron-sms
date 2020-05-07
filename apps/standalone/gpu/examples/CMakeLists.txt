if (WIN32)
	add_definitions(-D_USE_MATH_DEFINES)
endif()

add_executable(example examplemain.cpp)

install(TARGETS example DESTINATION bin COMPONENT main)
