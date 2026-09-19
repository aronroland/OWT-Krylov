#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ilu_audit {
using Real = long double;
using Vec = std::vector<Real>;
using Dense = std::vector<Vec>;
using Pattern = std::vector<std::vector<bool>>;

inline void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

inline Real dot(const Vec& a, const Vec& b)
{
    Real result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) result += a[i] * b[i];
    return result;
}

inline Vec multiply(const Dense& a, const Vec& x)
{
    Vec y(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) y[i] = dot(a[i], x);
    return y;
}

inline Vec residual(const Dense& a, const Vec& b, const Vec& x)
{
    auto r = multiply(a, x);
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = b[i] - r[i];
    return r;
}

inline void close(const Vec& actual, const Vec& expected, Real tolerance,
                  const std::string& name)
{
    require(actual.size() == expected.size(), name + ": size mismatch");
    Real scale = 0;
    for (Real x : expected) scale = std::max(scale, std::abs(x));
    scale = std::max(scale, Real(1e-30L));
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])
            || std::abs(actual[i] - expected[i]) > tolerance * scale) {
            std::ostringstream message;
            message.precision(18);
            message << name << " entry=" << i << " actual=" << actual[i]
                    << " expected=" << expected[i] << " scale=" << scale;
            throw std::runtime_error(message.str());
        }
    }
}

// Dot-product Doolittle construction, independent of the production CSR updates.
struct Factors {
    Dense lower, upper;
    Factors(const Dense& a, const Pattern& pattern)
        : lower(a.size(), Vec(a.size())), upper(a.size(), Vec(a.size()))
    {
        const auto n = a.size();
        for (std::size_t i = 0; i < n; ++i) {
            lower[i][i] = 1;
            for (std::size_t j = 0; j < i; ++j) {
                if (!pattern[i][j]) continue;
                Real value = a[i][j];
                for (std::size_t k = 0; k < j; ++k)
                    value -= lower[i][k] * upper[k][j];
                lower[i][j] = value / upper[j][j];
            }
            for (std::size_t j = i; j < n; ++j) {
                if (!pattern[i][j]) continue;
                Real value = a[i][j];
                for (std::size_t k = 0; k < i; ++k)
                    value -= lower[i][k] * upper[k][j];
                upper[i][j] = value;
            }
            require(std::isfinite(upper[i][i]) && upper[i][i] != 0,
                    "reference factorization: invalid pivot");
        }
    }

    Vec solve(const Vec& rhs) const
    {
        Vec x(rhs);
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = 0; j < i; ++j) x[i] -= lower[i][j] * x[j];
        for (std::size_t i = x.size(); i-- > 0;) {
            for (std::size_t j = i + 1; j < x.size(); ++j)
                x[i] -= upper[i][j] * x[j];
            x[i] /= upper[i][i];
        }
        return x;
    }

    Dense product() const
    {
        Dense p(lower.size(), Vec(lower.size()));
        for (std::size_t i = 0; i < p.size(); ++i)
            for (std::size_t j = 0; j < p.size(); ++j)
                for (std::size_t k = 0; k < p.size(); ++k)
                    p[i][j] += lower[i][k] * upper[k][j];
        return p;
    }
};

struct Iteration {
    Vec x, r, p, z, v, s, y, t;
    Real rho = 0, alpha = 1, beta = 0, omega = 1;
};

template<class ApplyInverse>
std::vector<Iteration> bicgstab(const Dense& a, const Vec& b, const Vec& initial,
                              ApplyInverse inverse, std::size_t count)
{
    std::vector<Iteration> history;
    const Vec shadow = residual(a, b, initial);
    Iteration state;
    state.x = initial;
    state.r = shadow;
    state.p = Vec(b.size());
    state.v = Vec(b.size());
    Real previous_rho = 1;
    for (std::size_t iteration = 0; iteration < count; ++iteration) {
        state.rho = dot(shadow, state.r);
        require(state.rho != 0, "reference BiCGSTAB rho breakdown");
        state.beta = iteration ? (state.rho / previous_rho)
                                     * (state.alpha / state.omega) : 0;
        for (std::size_t i = 0; i < b.size(); ++i)
            state.p[i] = state.r[i] + state.beta * (state.p[i] - state.omega * state.v[i]);
        state.z = inverse(state.p);
        state.v = multiply(a, state.z);
        state.alpha = state.rho / dot(shadow, state.v);
        state.s = state.r;
        for (std::size_t i = 0; i < b.size(); ++i) state.s[i] -= state.alpha * state.v[i];
        state.y = inverse(state.s);
        state.t = multiply(a, state.y);
        state.omega = dot(state.t, state.s) / dot(state.t, state.t);
        require(std::isfinite(state.omega) && state.omega != 0,
                "reference BiCGSTAB omega breakdown");
        for (std::size_t i = 0; i < b.size(); ++i) {
            state.x[i] += state.alpha * state.z[i] + state.omega * state.y[i];
            state.r[i] = state.s[i] - state.omega * state.t[i];
        }
        history.push_back(state);
        previous_rho = state.rho;
    }
    return history;
}
} // namespace ilu_audit
