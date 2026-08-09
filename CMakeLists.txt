cmake_minimum_required(VERSION 3.20)

project(OWTKrylov VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(owt_krylov INTERFACE)
add_library(OWT::Krylov ALIAS owt_krylov)
set_target_properties(owt_krylov PROPERTIES EXPORT_NAME Krylov)

target_compile_features(owt_krylov INTERFACE cxx_std_20)
target_include_directories(
    owt_krylov
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

option(OWT_KRYLOV_ENABLE_MPI "Enable MPI reductions and halo exchange" OFF)
option(OWT_KRYLOV_ENABLE_PETSC "Enable the optional PETSc adapter" OFF)
option(OWT_KRYLOV_ENABLE_LAPACK "Enable harmonic-Ritz GCRO-DR extraction" OFF)
option(OWT_KRYLOV_ENABLE_OPENMP_TARGET "Enable OpenMP Target execution policy" OFF)
option(OWT_KRYLOV_BUILD_TESTS "Build OWT-Krylov tests" ON)
option(OWT_KRYLOV_BUILD_BENCHMARKS "Build OWT-Krylov benchmark drivers" OFF)

if(OWT_KRYLOV_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_link_libraries(owt_krylov INTERFACE MPI::MPI_CXX)
    target_compile_definitions(owt_krylov INTERFACE OWT_KRYLOV_ENABLE_MPI=1)
endif()

if(OWT_KRYLOV_ENABLE_PETSC)
    if(NOT OWT_KRYLOV_ENABLE_MPI)
        message(FATAL_ERROR "OWT_KRYLOV_ENABLE_PETSC requires OWT_KRYLOV_ENABLE_MPI")
    endif()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc)
    target_link_libraries(owt_krylov INTERFACE PkgConfig::PETSC)
    target_compile_definitions(owt_krylov INTERFACE OWT_KRYLOV_ENABLE_PETSC=1)
endif()

if(OWT_KRYLOV_ENABLE_LAPACK)
    find_package(LAPACK REQUIRED)
    target_link_libraries(owt_krylov INTERFACE LAPACK::LAPACK)
    target_compile_definitions(owt_krylov INTERFACE OWT_KRYLOV_ENABLE_LAPACK=1)
endif()

if(OWT_KRYLOV_ENABLE_OPENMP_TARGET)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(owt_krylov INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(
        owt_krylov INTERFACE OWT_KRYLOV_ENABLE_OPENMP_TARGET=1)
endif()

if(OWT_KRYLOV_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(OWT_KRYLOV_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

install(
    TARGETS owt_krylov
    EXPORT OWTKrylovTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    EXPORT OWTKrylovTargets
    FILE OWTKrylovTargets.cmake
    NAMESPACE OWT::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OWTKrylov
)
configure_package_config_file(
    cmake/OWTKrylovConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OWTKrylovConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OWTKrylov
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OWTKrylovConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/OWTKrylovConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/OWTKrylovConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OWTKrylov
)
