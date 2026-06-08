// hpxpy::DenseMatrix — a minimal row-major 2-D float64 dense matrix, just enough for
// SpMM operands/results (sparse x dense -> dense). A contiguous dvec of rows*cols;
// element [i,j] is data[i*cols + j]. A focused down-payment on the (still deferred)
// general N-D story. Binding-agnostic, like array.hpp.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "array.hpp"    // dvec, on_hpx_thread

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hpxpy {

class DenseMatrix
{
public:
    DenseMatrix() = default;
    DenseMatrix(std::size_t rows, std::size_t cols, double fill)
      : rows_(rows), cols_(cols)
    {
        std::size_t const n = rows * cols;
        on_hpx_thread([&] { data_ = std::make_shared<dvec>(n, fill); });
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return rows_ * cols_; }

    double at(std::size_t i, std::size_t j) const
    {
        return data_->data()[i * cols_ + j];
    }
    void set(std::size_t i, std::size_t j, double v)
    {
        data_->data()[i * cols_ + j] = v;
    }

    // Contiguous row-major buffer (for kernels / the baseline).
    const double* data() const { return size() ? data_->data() : nullptr; }
    double* mutable_data() { return size() ? data_->data() : nullptr; }

    static DenseMatrix zeros(std::size_t rows, std::size_t cols)
    {
        return DenseMatrix(rows, cols, 0.0);
    }
    // Row-major fill from a flat list (rows*cols values).
    static DenseMatrix from_flat(std::size_t rows, std::size_t cols,
        std::vector<double> const& flat)
    {
        if (flat.size() != rows * cols)
            throw std::invalid_argument("dense_from: need rows*cols values");
        DenseMatrix m(rows, cols, 0.0);
        if (!flat.empty())
        {
            double* p = m.data_->data();
            for (std::size_t k = 0; k < flat.size(); ++k)
                p[k] = flat[k];
        }
        return m;
    }

private:
    std::size_t rows_ = 0, cols_ = 0;
    std::shared_ptr<dvec> data_;
};

inline DenseMatrix dense_zeros(std::size_t rows, std::size_t cols)
{
    return DenseMatrix::zeros(rows, cols);
}
inline DenseMatrix dense_from(std::size_t rows, std::size_t cols,
    std::vector<double> const& flat)
{
    return DenseMatrix::from_flat(rows, cols, flat);
}

}    // namespace hpxpy
