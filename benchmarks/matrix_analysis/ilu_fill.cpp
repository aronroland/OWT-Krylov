#include <owt/krylov/preconditioner.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>

template<class T> std::vector<T> read(std::ifstream& file, std::size_t count)
{
    std::vector<T> values(count);
    file.read(reinterpret_cast<char*>(values.data()),count*sizeof(T));
    return values;
}

template<class Range> void write(std::ofstream& file, const Range& values)
{
    file.write(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(values[0]));
}

int main(int argc, char** argv)
{
    try {
        if (argc!=4) throw std::invalid_argument("usage: ilu_fill INPUT OUTPUT LEVEL");
        std::ifstream input(argv[1],std::ios::binary);
        input.exceptions(std::ios::badbit|std::ios::failbit);
        const auto header=read<std::uint64_t>(input,3);
        const auto n=header[0], bins=header[1], nnz=header[2];
        if (!n || !bins || nnz<n) throw std::invalid_argument("invalid ILU fixture dimensions");
        const auto offsets=read<std::uint64_t>(input,n+1);
        const auto columns=read<std::uint64_t>(input,nnz);
        auto values=read<double>(input,nnz*bins);
        owt::krylov::BlockCsrMatrix<double,std::uint64_t> matrix(n,0,bins,offsets,columns,std::move(values));
        const auto start=std::chrono::steady_clock::now();
        owt::krylov::IluLevelPreconditioner<double,std::uint64_t> factor(matrix,std::stoul(argv[3]));
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ofstream output(argv[2],std::ios::binary);
        output.exceptions(std::ios::badbit|std::ios::failbit);
        write(output,std::vector<std::uint64_t>{n,bins,factor.factor_columns().size()});
        write(output,std::vector<std::uint64_t>(factor.factor_row_offsets().begin(),factor.factor_row_offsets().end()));
        write(output,std::vector<std::uint64_t>(factor.factor_columns().begin(),factor.factor_columns().end()));
        write(output,factor.factor_values());
        write(output,factor.inverse_diagonal());
        std::cout << "level=" << factor.fill_level() << " factor_node_entries=" << factor.factor_columns().size()
                  << " factor_value_bytes=" << factor.factor_values().size()*sizeof(double)
                  << " setup_seconds=" << seconds << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
