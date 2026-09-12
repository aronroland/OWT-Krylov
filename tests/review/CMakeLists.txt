add_executable(owt_review_cases review_cases.cpp)
target_link_libraries(owt_review_cases PRIVATE OWT::Krylov)
foreach(case IN ITEMS nonfinite_norm large_norm small_norm nonfinite_solve
        recycled_view candidate_view ssor scaled_gmres scaled_bicgstab ilu_pivot
        changed_recycle_operator final_iteration one_sync send_buffer)
    add_test(NAME review_${case} COMMAND owt_review_cases ${case})
    set_tests_properties(review_${case} PROPERTIES LABELS review TIMEOUT 15)
endforeach()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(owt_review_timer review_cases.cpp)
    target_link_libraries(owt_review_timer PRIVATE OWT::Krylov)
    target_compile_options(owt_review_timer PRIVATE -fno-elide-constructors)
    add_test(NAME review_timer_no_elision COMMAND owt_review_timer timer)
    set_tests_properties(review_timer_no_elision PROPERTIES LABELS review TIMEOUT 15)
endif()

if(OWT_KRYLOV_ENABLE_LAPACK)
    add_executable(owt_review_harmonic review_harmonic.cpp)
    target_link_libraries(owt_review_harmonic PRIVATE OWT::Krylov)
    foreach(case IN ITEMS flexible pair contracts)
        add_test(NAME review_harmonic_${case} COMMAND owt_review_harmonic ${case})
        set_tests_properties(review_harmonic_${case} PROPERTIES LABELS review TIMEOUT 15)
    endforeach()
endif()

if(OWT_KRYLOV_ENABLE_MPI)
    add_executable(owt_review_mpi review_mpi.cpp)
    target_link_libraries(owt_review_mpi PRIVATE OWT::Krylov)
    foreach(case IN ITEMS idr_global_dimension mixed_precision overlap_pattern)
        add_test(NAME review_mpi_${case}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
                    $<TARGET_FILE:owt_review_mpi> ${case})
        set_tests_properties(review_mpi_${case} PROPERTIES LABELS review TIMEOUT 15)
    endforeach()
endif()
