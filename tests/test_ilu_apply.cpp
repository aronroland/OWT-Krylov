#include <owt/krylov/preconditioner.hpp>

#include <bit>
#include <cstdint>
#include <iostream>
#include <numeric>

using namespace owt::krylov;

void same(std::span<const float> actual, std::span<const float> expected)
{
    if (actual.size() != expected.size()) throw std::runtime_error("size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]))
            throw std::runtime_error("component mismatch at " + std::to_string(i));
}

// Original two-vector application for bitwise comparison. Factor correctness
// is independently covered by the ILU audit.
void reference_ilu(const IluLevelPreconditioner<float>& pc,
                   const BlockVector<float>& input, BlockVector<float>& output)
{
    auto work = input.clone_layout();
    const auto size = input.block_size(), nodes = input.owned_nodes();
    auto rows = pc.factor_row_offsets(), columns = pc.factor_columns();
    auto factors = pc.factor_values(), inverse = pc.inverse_diagonal();
    for (std::size_t row = 0; row < nodes; ++row) {
        float* y = work.data() + row * size;
        std::copy_n(input.data() + row * size, size, y);
        for (auto entry = rows[row]; entry < rows[row + 1] && columns[entry] < row; ++entry)
            for (std::size_t c = 0; c < size; ++c)
                y[c] -= factors[entry * size + c] * work.data()[columns[entry] * size + c];
    }
    for (std::size_t row = nodes; row-- > 0;) {
        float* x = output.data() + row * size;
        std::copy_n(work.data() + row * size, size, x);
        for (auto entry = rows[row]; entry < rows[row + 1]; ++entry)
            if (columns[entry] > row)
                for (std::size_t c = 0; c < size; ++c)
                    x[c] -= factors[entry * size + c] * output.data()[columns[entry] * size + c];
        for (std::size_t c = 0; c < size; ++c) x[c] *= inverse[row * size + c];
    }
}

void check(std::size_t size)
{
    constexpr std::size_t nodes = 7, ghosts = 1;
    std::vector<std::size_t> offsets{0}, columns;
    std::vector<float> values;
    for (std::size_t row = 0; row < nodes; ++row) {
        // Unsorted rows, a nonlocal dependency and a pattern with ILU(1) fill.
        for (auto col : {row, (row + 2) % nodes, (row + nodes - 1) % nodes, nodes}) {
            columns.push_back(col);
            for (std::size_t c = 0; c < size; ++c)
                values.push_back(col == row ? 4.0F + float(c % 7) / 16.0F
                    : -0.03F * float(1 + (row + col + c) % 9));
        }
        offsets.push_back(columns.size());
    }
    BlockCsrMatrix<float> matrix(nodes, ghosts, size, offsets, columns, values);
    BlockVector<float> input(nodes, ghosts, size), actual(nodes, ghosts, size, 77.0F),
        expected(nodes, ghosts, size, 77.0F);
    for (std::size_t i = 0; i < input.local_size(); ++i)
        input.data()[i] = float(int(i % 29) - 14) / 11.0F;
    const std::vector<float> original(input.data(), input.data() + input.local_size());
    for (std::size_t fill : {0U, 1U, 2U}) {
        IluLevelPreconditioner<float> pc(matrix, fill);
        for (int update = 0; update < 2; ++update) {
            if (update) pc.update_values(matrix);
            reference_ilu(pc, input, expected);
            pc.apply(input, actual);
            same(actual.owned(), expected.owned());
            same({input.data(), input.local_size()}, original);
            if (!std::all_of(actual.ghosts().begin(), actual.ghosts().end(),
                             [](float x) { return x == 77.0F; }))
                throw std::runtime_error("ILU changed ghost values");
            auto alias = input.clone_layout();
            std::copy(original.begin(), original.end(), alias.data());
            pc.apply(alias, alias);
            same(alias.owned(), expected.owned());
            same(alias.ghosts(), input.ghosts());
        }
    }
}

int main()
{
    try {
        for (std::size_t size : {1U, 7U, 8U, 16U, 31U, 32U, 33U, 63U, 64U, 65U, 1260U, 1264U}) check(size);
        std::cout << "PASS: ILU apply matches the two-vector implementation bit-for-bit; wide blocks, ghosts, aliases and numeric updates\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
