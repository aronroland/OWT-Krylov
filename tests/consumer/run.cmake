foreach(required IN ITEMS OWT_BUILD OWT_SOURCE OWT_CONSUMER_ROOT)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}")
    endif()
endforeach()
string(TIMESTAMP run_id "%Y%m%dT%H%M%SZ" UTC)
set(OWT_CONSUMER_ROOT "${OWT_CONSUMER_ROOT}/runs/${run_id}")
if(EXISTS "${OWT_CONSUMER_ROOT}")
    message(FATAL_ERROR "Consumer evidence already exists: ${OWT_CONSUMER_ROOT}")
endif()
file(MAKE_DIRECTORY "${OWT_CONSUMER_ROOT}")
set(ENV{TMPDIR} "${OWT_CONSUMER_ROOT}")
set(ENV{OMPI_MCA_orte_tmpdir_base} "${OWT_CONSUMER_ROOT}")
set(ENV{PRTE_MCA_prte_tmpdir_base} "${OWT_CONSUMER_ROOT}")
set(ENV{OMPI_MCA_btl_vader_backing_directory} "${OWT_CONSUMER_ROOT}")

function(run label)
    file(WRITE "${OWT_CONSUMER_ROOT}/${label}.command" "${ARGN}\n")
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE status
        OUTPUT_FILE "${OWT_CONSUMER_ROOT}/${label}.log"
        ERROR_FILE "${OWT_CONSUMER_ROOT}/${label}.stderr")
    file(WRITE "${OWT_CONSUMER_ROOT}/${label}.exit" "${status}\n")
    if(NOT "${status}" STREQUAL "0")
        file(READ "${OWT_CONSUMER_ROOT}/${label}.log" output)
        file(READ "${OWT_CONSUMER_ROOT}/${label}.stderr" error)
        message(FATAL_ERROR "${label}: ${status}\n${output}\n${error}")
    endif()
endfunction()

run(install "${CMAKE_COMMAND}" --install "${OWT_BUILD}"
    --prefix "${OWT_CONSUMER_ROOT}/prefix")
run(configure "${CMAKE_COMMAND}" -S "${OWT_SOURCE}/tests/consumer"
    -B "${OWT_CONSUMER_ROOT}/build"
    "-DOWTKrylov_DIR=${OWT_CONSUMER_ROOT}/prefix/${OWT_INSTALL_LIBDIR}/cmake/OWTKrylov"
    "-DCMAKE_CXX_COMPILER=${OWT_CXX_COMPILER}"
    "-DCMAKE_CXX_FLAGS=${OWT_CXX_FLAGS}"
    "-DCMAKE_BUILD_TYPE=${OWT_BUILD_TYPE}"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
run(build "${CMAKE_COMMAND}" --build "${OWT_CONSUMER_ROOT}/build")
if(OWT_CXX_ID MATCHES "GNU|Clang|IntelLLVM")
    foreach(mode IN ITEMS fast finite)
        set(command "${CMAKE_COMMAND}" --build "${OWT_CONSUMER_ROOT}/build"
            --target unsafe_math_${mode})
        file(WRITE "${OWT_CONSUMER_ROOT}/reject-${mode}.command" "${command}\n")
        execute_process(COMMAND ${command} RESULT_VARIABLE status
            OUTPUT_FILE "${OWT_CONSUMER_ROOT}/reject-${mode}.log"
            ERROR_FILE "${OWT_CONSUMER_ROOT}/reject-${mode}.stderr")
        file(WRITE "${OWT_CONSUMER_ROOT}/reject-${mode}.exit" "${status}\n")
        file(READ "${OWT_CONSUMER_ROOT}/reject-${mode}.stderr" error)
        if("${status}" STREQUAL "0" OR NOT error MATCHES "OWT-Krylov requires finite-value checks")
            message(FATAL_ERROR "Unsafe math was not rejected by the library guard: ${mode}\n${error}")
        endif()
    endforeach()
endif()
run(ctest "${CMAKE_CTEST_COMMAND}" --test-dir "${OWT_CONSUMER_ROOT}/build" --output-on-failure)
