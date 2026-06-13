cmake_minimum_required(VERSION 3.12)
project(GEMV LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(HAVE_MLU "Enable MLU (Cambricon) support" ON)
option(HAVE_METAX "Enable MetaX support" OFF)
option(BUILD_TESTS "Build GEMV tests and autotuner launcher" ON)

if(HAVE_MLU AND HAVE_METAX)
    message(FATAL_ERROR "Cannot enable both MLU and MetaX simultaneously")
endif()

if(NOT HAVE_MLU AND NOT HAVE_METAX)
    message(FATAL_ERROR "Must enable either MLU or MetaX support. Use -DHAVE_MLU=ON or -DHAVE_METAX=ON")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/gemv.cpp
    src/utils/memory.cpp
    src/utils/handle.cpp
)

if(HAVE_MLU)
    message(STATUS "Building GEMV with MLU (Cambricon) support")
    set(NEUWARE_HOME "/usr/local/neuware" CACHE PATH "Cambricon Neuware installation path")
    set(MLU_ARCH 270 CACHE STRING "MLU target architecture, for example 270 or 592")
    set(GEMV_NRAM_CHUNK_FLOATS 256 CACHE STRING "NRAM chunk size in floats for GEMV MLU kernels")
    set(GEMV_TILE_SRAM_BLOCK_ROWS 16 CACHE STRING "SRAM row block size for tiled GEMV MLU kernels")
    set(GEMV_UNROLL_FACTOR 1 CACHE STRING "Tail scalar accumulation unroll factor for MLU GEMV kernels")
    set(CNCC_COMPILER "${NEUWARE_HOME}/bin/cncc" CACHE FILEPATH "Path to cncc compiler")
    set(GEMV_MLU_CONFIG_STAMP "${CMAKE_CURRENT_BINARY_DIR}/gemv_mlu_config.stamp")
    string(CONCAT GEMV_MLU_CONFIG_CONTENT
        "NRAM_CHUNK_FLOATS=${GEMV_NRAM_CHUNK_FLOATS}\n"
        "TILE_SRAM_BLOCK_ROWS=${GEMV_TILE_SRAM_BLOCK_ROWS}\n"
        "UNROLL_FACTOR=${GEMV_UNROLL_FACTOR}\n"
    )
    if(EXISTS "${GEMV_MLU_CONFIG_STAMP}")
        file(READ "${GEMV_MLU_CONFIG_STAMP}" GEMV_MLU_CONFIG_OLD)
    else()
        set(GEMV_MLU_CONFIG_OLD "")
    endif()
    if(NOT GEMV_MLU_CONFIG_OLD STREQUAL GEMV_MLU_CONFIG_CONTENT)
        file(WRITE "${GEMV_MLU_CONFIG_STAMP}" "${GEMV_MLU_CONFIG_CONTENT}")
    endif()

    message(STATUS "Using MLU architecture: mtp_${MLU_ARCH}")
    message(STATUS "Using cncc compiler: ${CNCC_COMPILER}")

    add_definitions(-DHAVE_MLU)
    include_directories(${NEUWARE_HOME}/include)
    link_directories(${NEUWARE_HOME}/lib64)

    file(GLOB_RECURSE MLU_KERNELS "src/kernels/mlu/*.mlu")
    foreach(kernel ${MLU_KERNELS})
        get_filename_component(kernel_name ${kernel} NAME_WE)
        file(RELATIVE_PATH kernel_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${kernel})
        get_filename_component(kernel_dir ${kernel_rel_path} DIRECTORY)
        string(REPLACE "/" "_" kernel_prefix ${kernel_dir})
        set(kernel_obj "${CMAKE_CURRENT_BINARY_DIR}/${kernel_prefix}_${kernel_name}.o")

        add_custom_command(
            OUTPUT ${kernel_obj}
            COMMAND ${CNCC_COMPILER}
                -c ${kernel}
                -o ${kernel_obj}
                -I${CMAKE_CURRENT_SOURCE_DIR}/include
                -I${NEUWARE_HOME}/include
                --bang-mlu-arch=mtp_${MLU_ARCH}
                -DNRAM_CHUNK_FLOATS=${GEMV_NRAM_CHUNK_FLOATS}
                -DTILE_SRAM_BLOCK_ROWS=${GEMV_TILE_SRAM_BLOCK_ROWS}
                -DUNROLL_FACTOR=${GEMV_UNROLL_FACTOR}
                -O2
            DEPENDS ${kernel} ${GEMV_MLU_CONFIG_STAMP}
            COMMENT "Compiling MLU kernel ${kernel_rel_path} with cncc"
            VERBATIM
        )
        list(APPEND KERNEL_OBJS ${kernel_obj})
    endforeach()
elseif(HAVE_METAX)
    message(STATUS "Building GEMV with MetaX support")
    find_package(MetaX REQUIRED)
    add_definitions(-DHAVE_METAX)

    file(GLOB_RECURSE METAX_KERNELS "src/kernels/metax/*.mx")
    foreach(kernel ${METAX_KERNELS})
        get_filename_component(kernel_name ${kernel} NAME_WE)
        file(RELATIVE_PATH kernel_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${kernel})
        get_filename_component(kernel_dir ${kernel_rel_path} DIRECTORY)
        string(REPLACE "/" "_" kernel_prefix ${kernel_dir})
        set(kernel_obj "${CMAKE_CURRENT_BINARY_DIR}/${kernel_prefix}_${kernel_name}.o")

        add_custom_command(
            OUTPUT ${kernel_obj}
            COMMAND metaxcc -c ${kernel} -o ${kernel_obj}
            DEPENDS ${kernel}
            COMMENT "Compiling MetaX kernel ${kernel_rel_path}"
        )
        list(APPEND KERNEL_OBJS ${kernel_obj})
    endforeach()
endif()

add_library(GEMV STATIC ${SOURCES} ${KERNEL_OBJS})
target_include_directories(GEMV PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(HAVE_MLU)
    target_link_libraries(GEMV PRIVATE cnrt)
    target_link_options(GEMV PRIVATE "-Wl,--no-eh-frame-hdr")
elseif(HAVE_METAX)
    target_link_libraries(GEMV PRIVATE MetaX::MetaX)
endif()

set_target_properties(GEMV PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_gemv tests/test_gemv.cpp)
    target_link_libraries(test_gemv PRIVATE GEMV)
    set_target_properties(test_gemv PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test
    )
    add_test(NAME test_gemv_quick COMMAND test_gemv quick)

    if(HAVE_MLU)
        add_executable(gemv_bo_launcher autotuner/launcher/gemv_bo_launcher.cpp)
        target_include_directories(gemv_bo_launcher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_compile_definitions(gemv_bo_launcher PRIVATE
            PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
            CNCC_COMPILER_PATH="${CNCC_COMPILER}"
        )
        target_link_libraries(gemv_bo_launcher PRIVATE GEMV cnrt dl)
        set_target_properties(gemv_bo_launcher PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
    endif()
endif()
