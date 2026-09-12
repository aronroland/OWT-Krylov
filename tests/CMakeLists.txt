add_executable(owt_krylov_core_tests test_core.cpp)
target_link_libraries(owt_krylov_core_tests PRIVATE OWT::Krylov)
target_compile_options(
    owt_krylov_core_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion>
)
add_test(NAME owt_krylov_core_tests COMMAND owt_krylov_core_tests)

add_executable(owt_krylov_split_ssor_tests test_split_ssor.cpp)
target_link_libraries(owt_krylov_split_ssor_tests PRIVATE OWT::Krylov)
target_compile_options(owt_krylov_split_ssor_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion>)
add_test(NAME owt_krylov_split_ssor_tests COMMAND owt_krylov_split_ssor_tests)
set_tests_properties(owt_krylov_split_ssor_tests PROPERTIES TIMEOUT 30)

add_executable(owt_krylov_foundation_tests test_foundation.cpp)
target_link_libraries(owt_krylov_foundation_tests PRIVATE OWT::Krylov)
foreach(case IN ITEMS norms nonfinite pipelined_view recycling)
    add_test(NAME owt_krylov_foundation_${case}
        COMMAND owt_krylov_foundation_tests ${case})
    set_tests_properties(owt_krylov_foundation_${case}
        PROPERTIES LABELS foundation TIMEOUT 30)
endforeach()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(owt_krylov_foundation_timer test_foundation.cpp)
    target_link_libraries(owt_krylov_foundation_timer PRIVATE OWT::Krylov)
    target_compile_options(owt_krylov_foundation_timer PRIVATE -fno-elide-constructors)
    add_test(NAME owt_krylov_foundation_timer
        COMMAND owt_krylov_foundation_timer timer)
    set_tests_properties(owt_krylov_foundation_timer
        PROPERTIES LABELS foundation TIMEOUT 30)
endif()

add_executable(owt_krylov_contract_tests review/review_contracts.cpp)
target_link_libraries(owt_krylov_contract_tests PRIVATE OWT::Krylov)
foreach(case IN ITEMS ssor scaled final ilu recycling)
    add_test(NAME owt_krylov_contract_${case} COMMAND owt_krylov_contract_tests ${case})
    set_tests_properties(owt_krylov_contract_${case} PROPERTIES LABELS contracts TIMEOUT 30)
endforeach()

option(OWT_KRYLOV_BUILD_REVIEW_TESTS "Build numerical review regression cases" OFF)
if(OWT_KRYLOV_BUILD_REVIEW_TESTS)
    add_subdirectory(review)
endif()

if(OWT_KRYLOV_ENABLE_MPI)
    add_executable(owt_krylov_mpi_contract_tests review/review_mpi_contracts.cpp)
    target_link_libraries(owt_krylov_mpi_contract_tests PRIVATE OWT::Krylov)
    foreach(case IN ITEMS idr mixed overlap halo)
        add_test(NAME owt_krylov_mpi_contract_${case}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    $<TARGET_FILE:owt_krylov_mpi_contract_tests> ${case})
        set_tests_properties(owt_krylov_mpi_contract_${case} PROPERTIES LABELS contracts TIMEOUT 30)
    endforeach()
    add_executable(owt_krylov_foundation_mpi test_foundation.cpp)
    target_link_libraries(owt_krylov_foundation_mpi PRIVATE OWT::Krylov)
    target_compile_definitions(owt_krylov_foundation_mpi PRIVATE OWT_FOUNDATION_MPI=1)
    foreach(case IN ITEMS norms nonfinite pipelined_view)
        add_test(NAME owt_krylov_foundation_mpi_${case}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    $<TARGET_FILE:owt_krylov_foundation_mpi> ${case})
        set_tests_properties(owt_krylov_foundation_mpi_${case}
            PROPERTIES LABELS foundation TIMEOUT 30)
    endforeach()
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
