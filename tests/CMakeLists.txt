add_executable(owt_krylov_core_tests test_core.cpp)
target_link_libraries(owt_krylov_core_tests PRIVATE OWT::Krylov)
target_compile_options(
    owt_krylov_core_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion>
)
add_test(NAME owt_krylov_core_tests COMMAND owt_krylov_core_tests)

add_executable(owt_krylov_ilu_apply test_ilu_apply.cpp)
target_link_libraries(owt_krylov_ilu_apply PRIVATE OWT::Krylov)
target_compile_options(owt_krylov_ilu_apply PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>)
add_test(NAME owt_krylov_ilu_apply COMMAND owt_krylov_ilu_apply)
set_tests_properties(owt_krylov_ilu_apply PROPERTIES LABELS contracts TIMEOUT 30)

add_executable(owt_krylov_schur_tests test_schur.cpp)
target_link_libraries(owt_krylov_schur_tests PRIVATE OWT::Krylov)
target_compile_options(owt_krylov_schur_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>)
add_test(NAME owt_krylov_schur_serial COMMAND owt_krylov_schur_tests)
set_tests_properties(owt_krylov_schur_serial PROPERTIES LABELS schur TIMEOUT 120)
if(OWT_KRYLOV_ENABLE_MPI)
    foreach(ranks IN ITEMS 2 4)
        add_test(NAME owt_krylov_schur_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:owt_krylov_schur_tests>)
        set_tests_properties(owt_krylov_schur_${ranks}_ranks PROPERTIES LABELS schur TIMEOUT 120)
    endforeach()
endif()

add_executable(owt_krylov_bicgstab_ilu_audit test_bicgstab_ilu_audit.cpp)
target_link_libraries(owt_krylov_bicgstab_ilu_audit PRIVATE OWT::Krylov)
target_compile_options(owt_krylov_bicgstab_ilu_audit PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>)
add_test(NAME owt_krylov_bicgstab_ilu_audit COMMAND owt_krylov_bicgstab_ilu_audit)
set_tests_properties(owt_krylov_bicgstab_ilu_audit PROPERTIES LABELS contracts TIMEOUT 30)

add_executable(owt_krylov_vertex_tests consumer/main.cpp)
target_link_libraries(owt_krylov_vertex_tests PRIVATE OWT::Krylov)
add_test(NAME owt_krylov_vertex_serial COMMAND owt_krylov_vertex_tests)
set_tests_properties(owt_krylov_vertex_serial PROPERTIES LABELS contracts TIMEOUT 120)
add_test(NAME owt_krylov_external_consumer
    COMMAND ${CMAKE_COMMAND}
        -DOWT_BUILD=${PROJECT_BINARY_DIR}
        -DOWT_SOURCE=${PROJECT_SOURCE_DIR}
        -DOWT_CONSUMER_ROOT=${PROJECT_BINARY_DIR}/consumer
        -DOWT_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
        -DOWT_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DOWT_CXX_ID=${CMAKE_CXX_COMPILER_ID}
        "-DOWT_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        -DOWT_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/consumer/run.cmake)
set_tests_properties(owt_krylov_external_consumer PROPERTIES LABELS contracts TIMEOUT 300)

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
    add_executable(owt_krylov_application_convergence test_application_convergence.cpp)
    target_link_libraries(owt_krylov_application_convergence PRIVATE OWT::Krylov)
    foreach(ranks IN ITEMS 1 2)
        add_test(NAME owt_krylov_application_convergence_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                    $<TARGET_FILE:owt_krylov_application_convergence>)
        set_tests_properties(owt_krylov_application_convergence_${ranks}_ranks
            PROPERTIES LABELS contracts TIMEOUT 120)
    endforeach()
    foreach(ranks IN ITEMS 2 4)
        add_test(NAME owt_krylov_vertex_${ranks}_ranks
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
                $<TARGET_FILE:owt_krylov_vertex_tests>)
        set_tests_properties(owt_krylov_vertex_${ranks}_ranks PROPERTIES LABELS contracts TIMEOUT 120)
    endforeach()
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
