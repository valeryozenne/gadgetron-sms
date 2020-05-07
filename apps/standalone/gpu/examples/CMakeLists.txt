if (WIN32)
	add_definitions(-D_USE_MATH_DEFINES)
endif()

add_executable(examples_run_gpu examplemain.cpp)

install(TARGETS examples_run_gpu DESTINATION bin COMPONENT main)
