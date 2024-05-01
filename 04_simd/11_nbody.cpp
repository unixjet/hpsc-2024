#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 x_all = _mm512_load_ps(x);
  __m512 y_all = _mm512_load_ps(y);
  __m512 m_all = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 x_i = _mm512_set1_ps(x[i]);
    __m512 y_i = _mm512_set1_ps(y[i]);

    __m512 rx = _mm512_sub_ps(x_i, x_all);
    __m512 ry = _mm512_sub_ps(y_i, y_all);

    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));

    // r_inv = sqrt(1/r^2) by using rsqrt14
    __m512 r_inv = _mm512_rsqrt14_ps(r2);

    __m512 r_cubed_inv = _mm512_mul_ps(_mm512_mul_ps(r_inv, r_inv), r_inv);

    __mmask16 mask = _cvtu32_mask16(~(1 << i));
    __m512 forces_x = _mm512_maskz_mul_ps(mask, rx, _mm512_mul_ps(m_all, r_cubed_inv));
    __m512 forces_y = _mm512_maskz_mul_ps(mask, ry, _mm512_mul_ps(m_all, r_cubed_inv));

    fx[i] -= _mm512_reduce_add_ps(forces_x);
    fy[i] -= _mm512_reduce_add_ps(forces_y);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }

  return 0;
}
