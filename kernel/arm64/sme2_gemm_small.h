/***************************************************************************
Copyright (c) 2026, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/


/* Register-blocked NEON GEMM for problems below the cost of streaming mode (column-major, no packing). */

#ifndef SME2_GEMM_SMALL_H
#define SME2_GEMM_SMALL_H

#include <arm_neon.h>
#ifdef DOUBLE
#define S2S_NV 2
typedef float64x2_t s2s_v;
#define S2S_LD vld1q_f64
#define S2S_ST vst1q_f64
#define S2S_DUP vdupq_n_f64
#define S2S_FMA_L(acc, a, b, l) vfmaq_laneq_f64(acc, a, b, l)
#define S2S_MUL vmulq_f64
#define S2S_FMA vfmaq_f64
#else
#define S2S_NV 4
typedef float32x4_t s2s_v;
#define S2S_LD vld1q_f32
#define S2S_ST vst1q_f32
#define S2S_DUP vdupq_n_f32
#define S2S_FMA_L(acc, a, b, l) vfmaq_laneq_f32(acc, a, b, l)
#define S2S_MUL vmulq_f32
#define S2S_FMA vfmaq_f32
#endif
#define S2S_MR (2 * S2S_NV)
#define S2S_NR 4

#define S2S_INL static inline __attribute__((always_inline))

S2S_INL s2s_v s2s_fmal(s2s_v acc, s2s_v a, s2s_v b, const int t) {
  switch (t) {
    case 0: return S2S_FMA_L(acc, a, b, 0);
#if S2S_NV > 2
    case 2: return S2S_FMA_L(acc, a, b, 2);
    case 3: return S2S_FMA_L(acc, a, b, 3);
#endif
    default: return S2S_FMA_L(acc, a, b, 1);
  }
}

/* C block of MV vectors x NR columns at (a, b, c); the first skip rows of the block are left unchanged in C. */
S2S_INL void s2s_blk(const int TB, const int MV, const int NR, int skip, BLASLONG k, const FLOAT *a, BLASLONG lda, const FLOAT *b,
                 BLASLONG ldb, FLOAT alpha, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  s2s_v acc[4][2];
  for (int j = 0; j < NR; ++j)
    for (int h = 0; h < MV; ++h) acc[j][h] = S2S_DUP(0);
  BLASLONG l = 0;
  if (!TB) /* B(l, j) at b[l + j * ldb]: S2S_NV depth steps per load of each column */
    for (; l + S2S_NV <= k; l += S2S_NV) {
      s2s_v bv[4];
      for (int j = 0; j < NR; ++j) bv[j] = S2S_LD(b + l + j * ldb);
#pragma clang loop unroll(full)
      for (int t = 0; t < S2S_NV; ++t) {
        s2s_v av[2];
        for (int h = 0; h < MV; ++h) av[h] = S2S_LD(a + (l + t) * lda + h * S2S_NV);
#pragma clang loop unroll(full)
        for (int j = 0; j < NR; ++j)
          for (int h = 0; h < MV; ++h) acc[j][h] = s2s_fmal(acc[j][h], av[h], bv[j], t);
      }
    }
  else if (NR == 4) /* B(l, j) at b[j + l * ldb]: the four columns of one depth step in 4 / S2S_NV loads */
    for (; l < k; ++l) {
      s2s_v av[2], bv[4 / S2S_NV];
      for (int h = 0; h < MV; ++h) av[h] = S2S_LD(a + l * lda + h * S2S_NV);
      for (int q = 0; q < 4 / S2S_NV; ++q) bv[q] = S2S_LD(b + l * ldb + q * S2S_NV);
#pragma clang loop unroll(full)
      for (int j = 0; j < 4; ++j)
        for (int h = 0; h < MV; ++h) acc[j][h] = s2s_fmal(acc[j][h], av[h], bv[j / S2S_NV], j % S2S_NV);
    }
  const BLASLONG sl = TB ? ldb : 1, sj = TB ? 1 : ldb;
  for (; l < k; ++l) {
    s2s_v av[2];
    for (int h = 0; h < MV; ++h) av[h] = S2S_LD(a + l * lda + h * S2S_NV);
    for (int j = 0; j < NR; ++j) {
      const s2s_v bj = S2S_DUP(b[l * sl + j * sj]);
      for (int h = 0; h < MV; ++h) acc[j][h] = S2S_FMA(acc[j][h], av[h], bj);
    }
  }
  const s2s_v va = S2S_DUP(alpha), vb = S2S_DUP(beta);
  for (int j = 0; j < NR; ++j) {
    FLOAT *cj = c + j * ldc;
    for (int h = 0; h < MV; ++h) {
      FLOAT *ch = cj + h * S2S_NV;
      const s2s_v r = beta == (FLOAT)0 ? S2S_MUL(acc[j][h], va) : S2S_FMA(S2S_MUL(S2S_LD(ch), vb), acc[j][h], va);
      if (skip == 0) S2S_ST(ch, r);
      else {
        FLOAT tmp[S2S_NV];
        S2S_ST(tmp, r);
        for (int i = skip; i < S2S_NV; ++i) ch[i] = tmp[i];
      }
    }
  }
}

/* Rows fewer than one vector: plain loops. */
static void s2s_rows(int tb, BLASLONG m, BLASLONG n, BLASLONG k, const FLOAT *a, BLASLONG lda, const FLOAT *b,
                   BLASLONG ldb, FLOAT alpha, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  const BLASLONG sl = tb ? ldb : 1, sj = tb ? 1 : ldb;
  for (BLASLONG j = 0; j < n; ++j)
    for (BLASLONG i = 0; i < m; ++i) {
      FLOAT s = 0;
      for (BLASLONG l = 0; l < k; ++l) s += a[i + l * lda] * b[l * sl + j * sj];
      c[i + j * ldc] = alpha * s + (beta == (FLOAT)0 ? (FLOAT)0 : beta * c[i + j * ldc]);
    }
}

/* One block column of NR columns: two-vector row blocks, one-vector blocks, and the row tail as an overlapped block. */
S2S_INL void s2s_col(const int TB, const int NR, BLASLONG m, BLASLONG k, const FLOAT *a, BLASLONG lda, const FLOAT *b, BLASLONG ldb,
                 FLOAT alpha, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  BLASLONG i = 0;
  for (; i + 2 * S2S_NV <= m; i += 2 * S2S_NV) s2s_blk(TB, 2, NR, 0, k, a + i, lda, b, ldb, alpha, beta, c + i, ldc);
  if (i + S2S_NV <= m) {
    s2s_blk(TB, 1, NR, 0, k, a + i, lda, b, ldb, alpha, beta, c + i, ldc);
    i += S2S_NV;
  }
  if (i < m) {
    if (m >= S2S_NV) s2s_blk(TB, 1, NR, (int)(S2S_NV - (m - i)), k, a + m - S2S_NV, lda, b, ldb, alpha, beta, c + m - S2S_NV, ldc);
    else s2s_rows(TB, m, NR, k, a, lda, b, ldb, alpha, beta, c, ldc);
  }
}

/* C = alpha A op(B) + beta C with A column-major; tb: B transposed. */
static void s2s_gemm_n(int tb, BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha, const FLOAT *a, BLASLONG lda, const FLOAT *b,
                     BLASLONG ldb, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  const BLASLONG sj = tb ? 1 : ldb;
  BLASLONG j = 0;
  if (tb)
    for (; j + 4 <= n; j += 4) s2s_col(1, 4, m, k, a, lda, b + j * sj, ldb, alpha, beta, c + j * ldc, ldc);
  else
    for (; j + 4 <= n; j += 4) s2s_col(0, 4, m, k, a, lda, b + j * sj, ldb, alpha, beta, c + j * ldc, ldc);
#define S2S_TAIL(NR)                                                                                                   \
  if (tb) s2s_col(1, NR, m, k, a, lda, b + j * sj, ldb, alpha, beta, c + j * ldc, ldc);                                \
  else s2s_col(0, NR, m, k, a, lda, b + j * sj, ldb, alpha, beta, c + j * ldc, ldc);
  switch (n - j) {
    case 3: S2S_TAIL(3) break;
    case 2: S2S_TAIL(2) break;
    case 1: S2S_TAIL(1) break;
  }
#undef S2S_TAIL
}

#ifdef DOUBLE
#define S2S_ADDV vaddvq_f64
#else
#define S2S_ADDV vaddvq_f32
#endif

/* C block of R rows x NR columns for C = alpha A^T B + beta C: dot products along the contiguous depth. */
S2S_INL void s2s_dot(const int R, const int NR, BLASLONG k, const FLOAT *a, BLASLONG lda, const FLOAT *b, BLASLONG ldb,
                 FLOAT alpha, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  s2s_v acc[4][4];
  for (int r = 0; r < R; ++r)
    for (int j = 0; j < NR; ++j) acc[r][j] = S2S_DUP(0);
  BLASLONG l = 0;
  for (; l + S2S_NV <= k; l += S2S_NV) {
    s2s_v av[4], bv[4];
    for (int r = 0; r < R; ++r) av[r] = S2S_LD(a + r * lda + l);
    for (int j = 0; j < NR; ++j) bv[j] = S2S_LD(b + j * ldb + l);
    for (int r = 0; r < R; ++r)
      for (int j = 0; j < NR; ++j) acc[r][j] = S2S_FMA(acc[r][j], av[r], bv[j]);
  }
  for (int r = 0; r < R; ++r)
    for (int j = 0; j < NR; ++j) {
      FLOAT s = S2S_ADDV(acc[r][j]);
      for (BLASLONG q = l; q < k; ++q) s += a[r * lda + q] * b[j * ldb + q];
      FLOAT *cc = c + r + j * ldc;
      *cc = alpha * s + (beta == (FLOAT)0 ? (FLOAT)0 : beta * *cc);
    }
}

S2S_INL void s2s_dot_rows(const int NR, BLASLONG m, BLASLONG k, const FLOAT *a, BLASLONG lda, const FLOAT *b, BLASLONG ldb,
                      FLOAT alpha, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  BLASLONG i = 0;
  for (; i + 4 <= m; i += 4) s2s_dot(4, NR, k, a + i * lda, lda, b, ldb, alpha, beta, c + i, ldc);
  switch (m - i) {
    case 3: s2s_dot(3, NR, k, a + i * lda, lda, b, ldb, alpha, beta, c + i, ldc); break;
    case 2: s2s_dot(2, NR, k, a + i * lda, lda, b, ldb, alpha, beta, c + i, ldc); break;
    case 1: s2s_dot(1, NR, k, a + i * lda, lda, b, ldb, alpha, beta, c + i, ldc); break;
  }
}

/* C = alpha A^T B + beta C. */
static void s2s_gemm_tn(BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha, const FLOAT *a, BLASLONG lda, const FLOAT *b,
                      BLASLONG ldb, FLOAT beta, FLOAT *c, BLASLONG ldc) {
  BLASLONG j = 0;
  for (; j + 4 <= n; j += 4) s2s_dot_rows(4, m, k, a, lda, b + j * ldb, ldb, alpha, beta, c + j * ldc, ldc);
  switch (n - j) {
    case 3: s2s_dot_rows(3, m, k, a, lda, b + j * ldb, ldb, alpha, beta, c + j * ldc, ldc); break;
    case 2: s2s_dot_rows(2, m, k, a, lda, b + j * ldb, ldb, alpha, beta, c + j * ldc, ldc); break;
    case 1: s2s_dot_rows(1, m, k, a, lda, b + j * ldb, ldb, alpha, beta, c + j * ldc, ldc); break;
  }
}

#endif
