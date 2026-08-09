add_executable(owt_krylov_core_tests test_core.cpp)
target_link_libraries(owt_krylov_core_tests PRIVATE OWT::Krylov)
target_compile_options(
    owt_krylov_core_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion>
)
add_test(NAME owt_krylov_core_tests COMMAND owt_krylov_core_tests)

if(OWT_KRYLOV_ENABLE_MPI)
    add_executable(owt_krylov_mpi_tests test_mpi.cpp)
    target_link_libraries(owt_krylov_mpi_tests PRIVATE OWT::Krylov)
    target_compile_options(
        owt_krylov_mpi_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    add_test(
        NAME owt_krylov_mpi_tests
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                $<TARGET_FILE:owt_krylov_mpi_tests>
    )

    add_executable(owt_krylov_partition_tests test_mpi_partitions.cpp)
    target_link_libraries(owt_krylov_partition_tests PRIVATE OWT::Krylov)
    target_compile_options(
        owt_krylov_partition_tests
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
    )
    foreach(ranks IN ITEMS 2 4)
        add_test(
            NAME owt_krylov_partition_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    $<TARGET_FILE:owt_krylov_partition_tests>
        )
    endforeach()
endif()
