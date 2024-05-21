find_package(stb)

add_executable(conv_example_1
    main.cpp)
target_link_libraries(conv_example_1 stb::stb)