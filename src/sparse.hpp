// hpxpy::CsrMatrix — a CSR (compressed sparse row) float64 matrix, binding-agnostic.
//
// Three HPX buffers: row_ptr (rows+1) and col_idx (nnz) as int64 compute::vectors,
// values (nnz) as the dense dvec. Kernels are thin wrappers over HPX parallel
// algorithms run DIRECTLY (the op-inlining lesson from array.hpp), so the
// abstraction penalty vs hand-written C++ HPX stays ~1.0. NWGraph-relevant.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "array.hpp"
#include "matrix.hpp"

#include <hpx/algorithm.hpp>
#include <hpx/execution.hpp>
#include <hpx/modules/compute_local.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace hpxpy {

using ivec =
    hpx::compute::vector<std::int64_t, hpx::compute::host::block_allocator<std::int64_t>>;

class CsrMatrix
{
public:
    CsrMatrix() = default;

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t nnz() const { return nnz_; }

    // Raw read-only buffer access (mirrors Array::data()) — for the C++ baseline and
    // the diag penalty ladder to run a direct kernel over the exact same CSR.
    const std::int64_t* row_ptr_data() const { return row_ptr_ ? row_ptr_->data() : nullptr; }
    const std::int64_t* col_idx_data() const { return nnz_ ? col_idx_->data() : nullptr; }
    const double* values_data() const { return nnz_ ? values_->data() : nullptr; }

    // y = A x. x has length cols_, y length rows_. Row-parallel: each row is an
    // independent sparse dot product. The for_loop is called directly (not nested
    // in a helper lambda) so the body inlines.
    Array spmv(Array const& x) const
    {
        if (x.size() != cols_)
            throw std::invalid_argument("spmv: x length must equal matrix cols");
        Array y(rows_, 0.0);
        if (rows_ != 0)
            spmv_into(x, y);
        return y;
    }

    // The bare kernel: write A·x into an EXISTING y (no allocation). Used by spmv()
    // and by the diagnostic to time the kernel in isolation from the result alloc.
    void spmv_into(Array const& x, Array& y) const
    {
        const std::int64_t* rp = row_ptr_->data();
        const std::int64_t* ci = nnz_ ? col_idx_->data() : nullptr;
        const double* vp = nnz_ ? values_->data() : nullptr;
        const double* xp = x.data();
        double* yp = y.mutable_data();
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), rows_,
            [rp, ci, vp, xp, yp](std::size_t i) {
                double acc = 0.0;
                for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k)
                    acc += vp[k] * xp[ci[k]];
                yp[i] = acc;
            });
    }

    // C = A B  (sparse x dense -> dense). B is cols_ x K, C is rows_ x K. Row-parallel:
    // each output row is a sparse combination of B's rows. spmm() allocates C; the bare
    // kernel spmm_into writes into an EXISTING C (no alloc) — used for clean kernel
    // timing (the result alloc otherwise dominates this memory-bound op, see SpMV).
    DenseMatrix spmm(DenseMatrix const& b) const
    {
        if (b.rows() != cols_)
            throw std::invalid_argument("spmm: B.rows must equal A.cols");
        DenseMatrix c(rows_, b.cols(), 0.0);
        if (rows_ != 0 && b.cols() != 0)
            spmm_into(b, c);
        return c;
    }

    void spmm_into(DenseMatrix const& b, DenseMatrix& c) const
    {
        std::size_t const K = b.cols();
        const std::int64_t* rp = row_ptr_->data();
        const std::int64_t* ci = nnz_ ? col_idx_->data() : nullptr;
        const double* vp = nnz_ ? values_->data() : nullptr;
        const double* bp = b.data();
        double* cp = c.mutable_data();
        hpx::experimental::for_loop(hpx::execution::par, std::size_t(0), rows_,
            [rp, ci, vp, bp, cp, K](std::size_t i) {
                double* crow = cp + i * K;
                for (std::size_t cc = 0; cc < K; ++cc)
                    crow[cc] = 0.0;
                for (std::int64_t k = rp[i]; k < rp[i + 1]; ++k)
                {
                    double const v = vp[k];
                    const double* brow = bp + static_cast<std::size_t>(ci[k]) * K;
                    for (std::size_t cc = 0; cc < K; ++cc)
                        crow[cc] += v * brow[cc];
                }
            });
    }

    // Build from explicit CSR arrays (copied into the HPX buffers).
    static CsrMatrix from_csr(std::size_t rows, std::size_t cols,
        std::vector<std::int64_t> const& row_ptr,
        std::vector<std::int64_t> const& col_idx,
        std::vector<double> const& values)
    {
        if (row_ptr.size() != rows + 1)
            throw std::invalid_argument("from_csr: row_ptr must have rows+1 entries");
        if (col_idx.size() != values.size())
            throw std::invalid_argument("from_csr: col_idx and values length mismatch");
        return build(rows, cols, row_ptr, col_idx, values);
    }

    // 1-D Laplacian: tridiagonal [-1, 2, -1], n x n. ~3 nnz/row; clean analytic
    // SpMV checks and a scalable memory-bound benchmark matrix.
    static CsrMatrix laplacian_1d(std::size_t n)
    {
        std::vector<std::int64_t> rp(n + 1, 0), ci;
        std::vector<double> vp;
        ci.reserve(3 * n);
        vp.reserve(3 * n);
        for (std::size_t i = 0; i < n; ++i)
        {
            if (i > 0) { ci.push_back(static_cast<std::int64_t>(i - 1)); vp.push_back(-1.0); }
            ci.push_back(static_cast<std::int64_t>(i)); vp.push_back(2.0);
            if (i + 1 < n) { ci.push_back(static_cast<std::int64_t>(i + 1)); vp.push_back(-1.0); }
            rp[i + 1] = static_cast<std::int64_t>(ci.size());
        }
        return build(n, n, rp, ci, vp);
    }

private:
    static CsrMatrix build(std::size_t rows, std::size_t cols,
        std::vector<std::int64_t> const& rp, std::vector<std::int64_t> const& ci,
        std::vector<double> const& vp)
    {
        CsrMatrix m;
        m.rows_ = rows;
        m.cols_ = cols;
        m.nnz_ = vp.size();
        on_hpx_thread([&] {
            m.row_ptr_ = std::make_shared<ivec>(rows + 1);
            m.col_idx_ = std::make_shared<ivec>(m.nnz_);
            m.values_ = std::make_shared<dvec>(m.nnz_);
            std::int64_t* rpd = m.row_ptr_->data();
            for (std::size_t i = 0; i <= rows; ++i)
                rpd[i] = rp[i];
            if (m.nnz_)
            {
                std::int64_t* cid = m.col_idx_->data();
                double* vpd = m.values_->data();
                for (std::size_t k = 0; k < m.nnz_; ++k)
                {
                    cid[k] = ci[k];
                    vpd[k] = vp[k];
                }
            }
        });
        return m;
    }

    std::size_t rows_ = 0, cols_ = 0, nnz_ = 0;
    std::shared_ptr<ivec> row_ptr_;
    std::shared_ptr<ivec> col_idx_;
    std::shared_ptr<dvec> values_;
};

inline CsrMatrix laplacian_1d(std::size_t n) { return CsrMatrix::laplacian_1d(n); }

}    // namespace hpxpy
