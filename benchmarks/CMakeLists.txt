add_executable(owt_krylov_unstructured_benchmark benchmark_unstructured.cpp)
target_link_libraries(owt_krylov_unstructured_benchmark PRIVATE OWT::Krylov)
target_compile_options(
    owt_krylov_unstructured_benchmark
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
)

if(OWT_KRYLOV_ENABLE_MPI)
    add_executable(owt_krylov_distributed_benchmark benchmark_distributed.cpp)
    target_link_libraries(owt_krylov_distributed_benchmark PRIVATE OWT::Krylov)
    target_compile_options(
        owt_krylov_distributed_benchmark
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-Wall;-Wextra;-Wpedantic>
    )
endif()
