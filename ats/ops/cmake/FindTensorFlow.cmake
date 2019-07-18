# Try to find TensorFlow and make sure we can use the tensorflow_framework
# library and include files.
# Once done it will define
# - TENSORFLOW_FOUND
# - TENSORFLOW_INCLUDE_DIR
# - TENSORFLOW_LIB

# Find the include dir first
execute_process(
    COMMAND python -c "import tensorflow as tf; \
        print(tf.sysconfig.get_include(), end='')"
    OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR
)
message(STATUS "Found TensorFlow include: " ${TENSORFLOW_INCLUDE_DIR})

# Find the library
execute_process(
    COMMAND python -c "import tensorflow as tf; \
        print(tf.sysconfig.get_lib(), end='')"
    OUTPUT_VARIABLE TENSORFLOW_LIB_DIR
)
execute_process(
    COMMAND python -c "import tensorflow as tf; \
        lib = next(f for f in tf.sysconfig.get_link_flags() \
                   if f.startswith('-l')); \
        print(lib[2:] if lib[2] != ':' else lib[3:], end='')"
    OUTPUT_VARIABLE TENSORFLOW_LIB_NAME
)
find_library(
    TENSORFLOW_LIB ${TENSORFLOW_LIB_NAME}
    PATHS ${TENSORFLOW_LIB_DIR}
)
message(STATUS "Found TensorFlow lib: " ${TENSORFLOW_LIB})

# Check for issue #27067 in TensorFlow. Basically make sure that if we have
# version > 1.13 we are using gcc < 5.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (NOT (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5))
        execute_process(
            COMMAND python -c "import tensorflow as tf; \
                print(tf.__version__, end='')"
            OUTPUT_VARIABLE TF_VERSION
        )
        if(NOT (TF_VERSION VERSION_LESS 1.14))
            if(DEFINED ENV{TF_COMPILER_VERSION_OVERRIDE})
                message(WARNING "You are using an unsupported compiler! \
                                 You have manually overriden our check and \
                                 your extension might break in undefined \
                                 ways. To remove the override unset the \
                                 TF_COMPILER_VERSION_OVERRIDE environment \
                                 variable")
            else()
                message(FATAL_ERROR "You are using an unsupported compiler! \
                                     TensorFlow extensions for versions >= 1.14 \
                                     must be build with gcc < 5. To override set \
                                     TF_COMPILER_VERSION_OVERRIDE environment \
                                     variable.")
            endif()
        else()
            message(WARNING "You are using an unsupported compiler! It is \
                             recommended to build your TensorFlow extensions \
                             with gcc 4.8 .")
        endif()
    endif()
else()
    message(WARNING "Unless you are using a custom built TensorFlow it \
                     is suggested to use gcc to compile your extensions")
endif()

# Add the CXX flags
execute_process(
    COMMAND python -c "import tensorflow as tf; \
        print(' '.join( \
            f for f in tf.sysconfig.get_compile_flags() \
            if not f.startswith('-I')), end='')"
    OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS
)
set(CMAKE_CXX_FLAGS "${TENSORFLOW_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
message(STATUS "Added TensorFlow flags: " ${TENSORFLOW_COMPILE_FLAGS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorFlow
    DEFAULT_MSG
    TENSORFLOW_INCLUDE_DIR
    TENSORFLOW_LIB
)
