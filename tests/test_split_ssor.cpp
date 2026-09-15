#include "../benchmarks/reference_split_ssor.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>

using namespace owt::krylov;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void same_owned(const BlockVector<T>& a, const BlockVector<T>& b)
{
    for (std::size_t i = 0; i < a.owned_size(); ++i) {
        using Bits = std::conditional_t<std::same_as<T, float>, std::uint32_t, std::uint64_t>;
        require(std::bit_cast<Bits>(a.data()[i]) == std::bit_cast<Bits>(b.data()[i]),
                "SSOR result differs bitwise from the frozen reference");
    }
}

template<class T>
void check(std::size_t rows, std::size_t block, T omega)
{
    constexpr std::size_t ghosts = 2;
    std::vector<std::uint32_t> offsets{0}, columns;
    std::vector<T> diagonal(rows * block, T(4)), edges;
    for (std::size_t row = 0; row < rows; ++row) {
        // Deliberately unsorted entries, including duplicates and ghost edges.
        auto add = [&](std::size_t column, T value) {
            columns.push_back(static_cast<std::uint32_t>(column));
            for (std::size_t c = 0; c < block; ++c)
                edges.push_back(value * (T(1) + T(c % 5) * T(0.01)));
        };
        add(rows + 1, std::numeric_limits<T>::quiet_NaN());
        if (row + 1 < rows) add(row + 1, T(-0.8));
        if (row > 0) add(row - 1, T(-1.2));
        if (row + 1 < rows) add(row + 1, T(0.1));
        offsets.push_back(static_cast<std::uint32_t>(columns.size()));
    }
    SplitBlockCsrMatrixView<T, std::uint32_t> matrix(rows, ghosts, block, offsets, columns, diagonal, edges);
    SplitLocalSsorPreconditioner<T, std::uint32_t> native(matrix, omega);
    benchmark::ReferenceSplitSsor<T, std::uint32_t> reference(matrix, omega);
    BlockVector<T> rhs(rows, ghosts, block);
    rhs.fill(std::numeric_limits<T>::quiet_NaN());
    for (std::size_t i = 0; i < rhs.owned_size(); ++i) rhs.data()[i] = T(std::sin(double(i)));
    std::vector<T> storage(rhs.local_size() + 2, T(77));
    auto output = BlockVector<T>::view(rows, ghosts, block, std::span<T>(storage).subspan(1, rhs.local_size()));
    auto expected = rhs.clone_layout();
    for (int repetition = 0; repetition < 3; ++repetition) {
        output.fill_owned(std::numeric_limits<T>::quiet_NaN());
        native.apply(rhs, output);
        reference.apply(rhs, expected);
        same_owned(output, expected);
        for (T value : output.owned()) require(std::isfinite(value), "SSOR read stale output or a ghost");
        for (std::size_t i = output.owned_size(); i < output.local_size(); ++i)
            require(output.data()[i] == T(77), "SSOR modified output ghosts");
        require(storage.front() == T(77) && storage.back() == T(77), "view guard was modified");
        copy_owned(rhs, output);
        native.apply(output, output);
        same_owned(output, expected);
        // Borrowed coefficient changes must remain immediately observable.
        for (T& value : diagonal) value += T(0.25);
    }
    native.update_values(matrix);
    reference.update_values(matrix);
    require(native.numeric_updates() == 2, "numeric update count changed");
    native.apply(rhs, output);
    reference.apply(rhs, expected);
    same_owned(output, expected);
    BlockVector<T> wrong(rows + 1, ghosts, block);
    bool rejected = false;
    try { native.apply(rhs, wrong); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "wrong output layout accepted");
}

template<class T>
void check_overlapping_views(std::size_t block, T omega)
{
    constexpr std::size_t rows = 7, ghosts = 2;
    const std::array<std::uint32_t, rows + 1> offsets{0, 3, 6, 9, 12, 15, 18, 21};
    std::vector<std::uint32_t> columns;
    std::vector<T> diagonal(rows * block), edges;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column : {(row + 1) % rows, rows + 1, (row + rows - 1) % rows}) {
            columns.push_back(static_cast<std::uint32_t>(column));
            for (std::size_t c = 0; c < block; ++c)
                edges.push_back(column >= rows ? std::numeric_limits<T>::quiet_NaN()
                    : T(-0.25) - T(c % 5) / T(32));
        }
    }
    for (std::size_t i = 0; i < diagonal.size(); ++i)
        diagonal[i] = T(3) + T(i % 7) / T(8);
    SplitBlockCsrMatrixView<T, std::uint32_t> matrix(rows, ghosts, block, offsets, columns, diagonal, edges);
    SplitLocalSsorPreconditioner<T, std::uint32_t> native(matrix, omega);
    benchmark::ReferenceSplitSsor<T, std::uint32_t> reference(matrix, omega);
    BlockVector<T> expected(rows, ghosts, block);
    const std::size_t owned = rows * block, local = (rows + ghosts) * block;
    for (std::size_t shift : {std::size_t(0), std::size_t(1), block, owned - 1, owned, local}) {
        for (bool output_first : {false, true}) {
            std::vector<T> storage(2 * local + 2, T(77));
            const std::size_t input_start = 1 + (output_first ? shift : 0);
            const std::size_t output_start = 1 + (output_first ? 0 : shift);
            auto input = BlockVector<T>::view(rows, ghosts, block,
                std::span<T>(storage).subspan(input_start, local));
            auto output = BlockVector<T>::view(rows, ghosts, block,
                std::span<T>(storage).subspan(output_start, local));
            for (std::size_t i = 0; i < owned; ++i)
                input.data()[i] = T(1) + T(i % 13) / T(16);
            std::fill(input.ghosts().begin(), input.ghosts().end(), std::numeric_limits<T>::quiet_NaN());
            const auto before = storage;
            reference.apply(input, expected);
            native.apply(input, output);
            same_owned(output, expected);
            for (std::size_t i = 0; i < storage.size(); ++i) {
                if (i >= output_start && i < output_start + owned) continue;
                using Bits = std::conditional_t<std::same_as<T, float>, std::uint32_t, std::uint64_t>;
                require(std::bit_cast<Bits>(storage[i]) == std::bit_cast<Bits>(before[i]),
                        "SSOR modified storage outside the owned output view");
            }
        }
    }
}
}

int main()
{
    try {
        for (std::size_t rows : {1U, 7U})
            for (std::size_t block : {1U, 5U, 1296U}) {
                for (float omega : {0.5F, 1.0F, std::nextafter(1.0F, 2.0F), 1.4F}) check<float>(rows, block, omega);
                for (double omega : {0.5, 1.0, std::nextafter(1.0, 2.0), 1.4}) check<double>(rows, block, omega);
            }
        for (std::size_t block : {1U, 5U, 31U, 32U, 33U, 1296U}) {
            for (float omega : {0.5F, 1.0F, std::nextafter(1.0F, 2.0F), 1.4F})
                check_overlapping_views<float>(block, omega);
            for (double omega : {0.5, 1.0, std::nextafter(1.0, 2.0), 1.4})
                check_overlapping_views<double>(block, omega);
        }
        std::cout << "PASS: split SSOR frozen-reference and borrowed-storage contracts\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
