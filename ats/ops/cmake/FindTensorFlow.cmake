# Try to find TensorFlow and make sure we can use the tensorflow_framework
# library and include files.
# Once done it will define
# - TENSORFLOW_FOUND
# - TENSORFLOW_INCLUDE_DIR
# - TENSORFLOW_LIB

execute_process(
    COMMAND python -c "import tensorflow as tf; \
        print(tf.sysconfig.get_include(), end='')"
    OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR
)
execute_process(
    COMMAND python -c "import tensorflow as tf; \
        print(tf.sysconfig.get_lib(), end='')"
    OUTPUT_VARIABLE TENSORFLOW_LIB_DIR
)
find_library(
    TENSORFLOW_LIB tensorflow_framework
    PATHS ${TENSORFLOW_LIB_DIR}
)
execute_process(
    COMMAND python -c "import tensorflow as tf; \
        print(' '.join( \
            f for f in tf.sysconfig.get_compile_flags() \
            if not f.startswith('-I')), end='')"
    OUTPUT_VARIABLE TENSORFLOW_COMPILE_FLAGS
)
set(CMAKE_CXX_FLAGS "${TENSORFLOW_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorFlow
    DEFAULT_MSG
    TENSORFLOW_INCLUDE_DIR
    TENSORFLOW_LIB
)
