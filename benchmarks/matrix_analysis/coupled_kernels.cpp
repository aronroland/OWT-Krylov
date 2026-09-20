#include <cmath>
#include <cstdint>

using Index = std::uint64_t;

// Array extents and index ranges are checked by the Python snapshot reader.
template<class Scalar>
void action(Index rows, Index bins, const Index* nodes, const Index* ptr,
            const Index* col, const Scalar* val, const double* x,
            double* y, int mode)
{
    for (Index i = 0; i < rows; ++i) {
        const Index row = nodes[i / bins] * bins + i % bins;
        double sum = 0;
        for (Index e = ptr[i]; e < ptr[i + 1]; ++e) {
            if (mode >= 2) y[col[e]] += (mode == 3 ? std::abs(double(val[e])) : double(val[e])) * x[row];
            else sum += (mode == 1 ? std::abs(double(val[e])) : double(val[e])) * x[col[e]];
        }
        if (mode < 2) y[row] = sum;
    }
}

template<class Scalar>
void factor(Index n, Index bins, const Index* nodes, const Index* ptr,
            const Index* col, const Scalar* val, const Scalar* inv,
            const double* x, double* y, int mode)
{
    for (Index i = 0; i < n; ++i)
        for (Index c = 0; c < bins; ++c)
            y[nodes[i] * bins + c] = x[nodes[i] * bins + c];
    if (mode == 0) {
        for (Index i = 0; i < n; ++i) {
            auto* yi = y + nodes[i] * bins;
            for (Index e = ptr[i]; e < ptr[i + 1]; ++e) if (col[e] < i)
                for (Index c = 0; c < bins; ++c)
                    yi[c] -= double(val[e * bins + c]) * y[nodes[col[e]] * bins + c];
        }
        for (Index k = n; k > 0; --k) {
            const Index i = k - 1;
            auto* yi = y + nodes[i] * bins;
            for (Index e = ptr[i]; e < ptr[i + 1]; ++e) if (col[e] > i)
                for (Index c = 0; c < bins; ++c)
                    yi[c] -= double(val[e * bins + c]) * y[nodes[col[e]] * bins + c];
            for (Index c = 0; c < bins; ++c) yi[c] *= double(inv[i * bins + c]);
        }
    } else if (mode == 1) {
        // P^-T = L^-T U^-T. Scatter updates avoid storing transpose factors.
        for (Index i = 0; i < n; ++i) {
            auto* yi = y + nodes[i] * bins;
            for (Index c = 0; c < bins; ++c) yi[c] *= double(inv[i * bins + c]);
            for (Index e = ptr[i]; e < ptr[i + 1]; ++e) if (col[e] > i)
                for (Index c = 0; c < bins; ++c)
                    y[nodes[col[e]] * bins + c] -= double(val[e * bins + c]) * yi[c];
        }
        for (Index k = n; k > 0; --k) {
            const Index i = k - 1;
            const auto* yi = y + nodes[i] * bins;
            for (Index e = ptr[i]; e < ptr[i + 1]; ++e) if (col[e] < i)
                for (Index c = 0; c < bins; ++c)
                    y[nodes[col[e]] * bins + c] -= double(val[e * bins + c]) * yi[c];
        }
    } else {
        for (Index i = 0; i < n; ++i) {
            auto* yi = y + nodes[i] * bins;
            for (Index c = 0; c < bins; ++c) yi[c] = x[nodes[i] * bins + c] / double(inv[i * bins + c]);
            for (Index e = ptr[i]; e < ptr[i + 1]; ++e) if (col[e] > i)
                for (Index c = 0; c < bins; ++c)
                    yi[c] += double(val[e * bins + c]) * x[nodes[col[e]] * bins + c];
        }
        // Reverse order keeps the unmodified Ux values available below i.
        for (Index k = n; k > 0; --k) {
            const Index i = k - 1;
            auto* yi = y + nodes[i] * bins;
            for (Index e = ptr[i]; e < ptr[i + 1]; ++e) if (col[e] < i)
                for (Index c = 0; c < bins; ++c)
                    yi[c] += double(val[e * bins + c]) * y[nodes[col[e]] * bins + c];
        }
    }
}

template<class Scalar>
void parts(Index rows, Index bins, Index nd, Index rank, const Index* owners,
           const Index* nodes, const Index* ptr, const Index* col,
           const Scalar* val, const unsigned char* kind, const double* x, double* y)
{
    for (Index i = 0; i < rows; ++i) {
        const Index row = nodes[i / bins] * bins + i % bins;
        for (Index e = ptr[i]; e < ptr[i + 1]; ++e) {
            const Index j = col[e];
            const int part = owners[j / bins] != rank ? 0 :
                (j % bins) / nd != (i % bins) / nd ? 1 :
                j % bins != i % bins ? 2 :
                kind[i] == 2 && j != row ? 3 : 4;
            y[5 * row + part] += double(val[e]) * x[j];
        }
    }
}

extern "C" {
void matrix_action(Index rows, Index bins, const Index* nodes, const Index* ptr,
                   const Index* col, const void* val, int bytes,
                   const double* x, double* y, int mode)
{
    if (bytes == 4) action(rows, bins, nodes, ptr, col, static_cast<const float*>(val), x, y, mode);
    else action(rows, bins, nodes, ptr, col, static_cast<const double*>(val), x, y, mode);
}

void factor_action(Index n, Index bins, const Index* nodes, const Index* ptr,
                   const Index* col, const void* val, const void* inv, int bytes,
                   const double* x, double* y, int mode)
{
    if (bytes == 4) factor(n, bins, nodes, ptr, col, static_cast<const float*>(val),
                          static_cast<const float*>(inv), x, y, mode);
    else factor(n, bins, nodes, ptr, col, static_cast<const double*>(val),
                static_cast<const double*>(inv), x, y, mode);
}

void matrix_parts(Index rows, Index bins, Index nd, Index rank, const Index* owners,
                  const Index* nodes, const Index* ptr, const Index* col, const void* val,
                  int bytes, const unsigned char* kind, const double* x, double* y)
{
    if (bytes == 4) parts(rows, bins, nd, rank, owners, nodes, ptr, col,
                         static_cast<const float*>(val), kind, x, y);
    else parts(rows, bins, nd, rank, owners, nodes, ptr, col,
               static_cast<const double*>(val), kind, x, y);
}
}
