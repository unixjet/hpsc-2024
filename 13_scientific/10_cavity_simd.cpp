#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <immintrin.h>

#define SIMD_ACCURACY_CHECK

using namespace std;
typedef vector<vector<float>> matrix;

#if defined(SIMD_ACCURACY_CHECK)

float l2_norm(matrix& v1, matrix& v2, int ny, int nx) {

  float sum = 0.0f;
  __m512 sum_vec = _mm512_setzero_ps();

  for(int j = 0; j < ny; j++) {
    int i = 0;
    for(; i < nx - 16; i += 16) {
      __m512 v1_vec = _mm512_loadu_ps(&v1[j][i]);
      __m512 v2_vec = _mm512_loadu_ps(&v2[j][i]);
      __m512 diff = _mm512_sub_ps(v1_vec, v2_vec);
      __m512 diff_sq = _mm512_mul_ps(diff, diff);
      sum_vec = _mm512_add_ps(sum_vec, diff_sq);
    }

    for(; i < nx; i++) {
      float diff = v1[j][i] - v2[j][i];
      sum += diff * diff;
    }
  }

  sum += _mm512_reduce_add_ps(sum_vec);
  return sqrt(sum);
}
#endif


int main() {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  matrix u(ny, vector<float>(nx, 0));
  matrix v(ny, vector<float>(nx, 0));
  matrix p(ny, vector<float>(nx, 0));
  matrix b(ny, vector<float>(nx, 0));
  matrix un(ny, vector<float>(nx, 0));
  matrix vn(ny, vector<float>(nx, 0));
  matrix pn(ny, vector<float>(nx, 0));

#if defined(SIMD_ACCURACY_CHECK)
  // Prefix r_ (reference)
  matrix r_u(ny, vector<float>(nx, 0));
  matrix r_v(ny, vector<float>(nx, 0));
  matrix r_p(ny, vector<float>(nx, 0));
  matrix r_b(ny, vector<float>(nx, 0));
  matrix r_un(ny, vector<float>(nx, 0));
  matrix r_vn(ny, vector<float>(nx, 0));
  matrix r_pn(ny, vector<float>(nx, 0));
#endif

  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");

#if defined(SIMD_ACCURACY_CHECK)
  printf("Results by SIMD will be verified.\n"
         "r_u, r_v and r_p are derived from CPU, whereas u, v and p are from SIMD.\n"
         "||x|| represents l2 norm.\n");
#endif

  for (int n = 0; n < nt; n++) {
    // Compute b
    for (int j = 1; j < ny - 1; j++) {
#define compute_b(xlen) \
      do { \
        __m##xlen u_right = _mm##xlen##_loadu_ps(&u[j][i + 1]); \
        __m##xlen u_left = _mm##xlen##_loadu_ps(&u[j][i - 1]); \
        __m##xlen u_top = _mm##xlen##_loadu_ps(&u[j - 1][i]); \
        __m##xlen u_bottom = _mm##xlen##_loadu_ps(&u[j + 1][i]); \
        \
        __m##xlen v_right = _mm##xlen##_loadu_ps(&v[j][i + 1]); \
        __m##xlen v_left = _mm##xlen##_loadu_ps(&v[j][i - 1]); \
        __m##xlen v_top = _mm##xlen##_loadu_ps(&v[j - 1][i]); \
        __m##xlen v_bottom = _mm##xlen##_loadu_ps(&v[j + 1][i]); \
        \
        __m##xlen du_dx = _mm##xlen##_div_ps(_mm##xlen##_sub_ps(u_right, u_left), _mm##xlen##_set1_ps(2 * dx));\
        __m##xlen dv_dy = _mm##xlen##_div_ps(_mm##xlen##_sub_ps(v_bottom, v_top), _mm##xlen##_set1_ps(2 * dy));\
        \
        __m##xlen du_dy = _mm##xlen##_div_ps(_mm##xlen##_sub_ps(u_bottom, u_top), _mm##xlen##_set1_ps(2 * dy));\
        __m##xlen dv_dx = _mm##xlen##_div_ps(_mm##xlen##_sub_ps(v_right, v_left), _mm##xlen##_set1_ps(2 * dx));\
        \
        __m##xlen term1 = _mm##xlen##_mul_ps(du_dx, du_dx);\
        __m##xlen term2 = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(2), _mm##xlen##_mul_ps(du_dy, dv_dx));\
        __m##xlen term3 = _mm##xlen##_mul_ps(dv_dy, dv_dy);\
        \
        __m##xlen t1 = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(1./ dt), _mm##xlen##_add_ps(du_dx, dv_dy));\
        __m##xlen t2 = _mm##xlen##_add_ps(term1, _mm##xlen##_add_ps(term2, term3));\
        __m##xlen b_res = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(rho), _mm##xlen##_sub_ps(t1, t2));\
        \
        _mm##xlen##_storeu_ps(&b[j][i], b_res);\
      }\
      while(0)

      int i = 1;
      for (; i < nx - 17; i += 16) {
        // Employ mm512
        compute_b(512);
      }

      // It seems never enter the below block when nx = 41.
      // Cauze the above block has handled [1-16](batch1) and [17-32](batch2]
      // with the leftover [33-39] (i.e. 7 is left exactly).
      // The block is kept here for general purpose. For example, when nx = 42;
      for (; i < nx - 9; i += 8) {
        // Employ mm256
        compute_b(256);
      }
#undef compute_b

      // Handle the remains
      for (; i < nx - 1; i++) {
        b[j][i] = rho * (1 / dt *
                  ((u[j][i + 1] - u[j][i - 1]) / (2 * dx) + (v[j + 1][i] - v[j - 1][i]) / (2 * dy)) -
                  pow((u[j][i + 1] - u[j][i - 1]) / (2 * dx), 2) -
                  2 * ((u[j + 1][i] - u[j - 1][i]) / (2 * dy) * (v[j][i + 1] - v[j][i - 1]) / (2 * dx)) -
                  pow((v[j + 1][i] - v[j - 1][i]) / (2 * dy), 2));
      }
    }

    // Compute p
    for (int it = 0; it < nit; it++) {
      // Copy p to pn
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          pn[j][i] = p[j][i];
        }
      }

      for (int j = 1; j < ny - 1; j++) {
#define compute_p(xlen)\
        do {\
          __m##xlen pn_left = _mm##xlen##_loadu_ps(&pn[j][i - 1]); \
          __m##xlen pn_right = _mm##xlen##_loadu_ps(&pn[j][i + 1]); \
          __m##xlen pn_bottom = _mm##xlen##_loadu_ps(&pn[j + 1][i]);\
          __m##xlen pn_top = _mm##xlen##_loadu_ps(&pn[j - 1][i]); \
          __m##xlen b_curr = _mm##xlen##_loadu_ps(&b[j][i]); \
          \
          __m##xlen term1 = _mm##xlen##_add_ps(pn_left, pn_right);\
          __m##xlen term2 = _mm##xlen##_add_ps(pn_bottom, pn_top);\
          __m##xlen term3 = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dy * dy), term1);\
          __m##xlen term4 = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dx * dx), term2); \
          __m##xlen b_term = _mm##xlen##_mul_ps(b_curr, _mm##xlen##_set1_ps(dx * dx * dy * dy));\
          \
          __m##xlen p_new = _mm##xlen##_div_ps(_mm##xlen##_sub_ps(_mm##xlen##_add_ps(term3, term4), b_term),\
                                       _mm##xlen##_set1_ps(2 * (dx * dx + dy * dy)));\
          \
          _mm##xlen##_storeu_ps(&p[j][i], p_new);\
        }while(0)

        int i = 1;
        for (; i < nx - 17; i += 16) {
          compute_p(512);
        }
        for (; i < nx - 9; i += 8) {
          compute_p(256);
        }
#undef compute_p

        // Handle remaining elements
        for (; i < nx - 1; i++) {
          p[j][i] = ((dy * dy) * (pn[j][i + 1] + pn[j][i - 1]) +
                     (dx * dx) * (pn[j + 1][i] + pn[j - 1][i]) -
                     b[j][i] * dx * dx * dy * dy) /
                    (2 * (dx * dx + dy * dy));
        }
      }

      // Boundary conditions for pressure
      for (int j = 0; j < ny; j++) {
        p[j][0] = p[j][1];
        p[j][nx - 1] = p[j][nx - 2];
      }
      for (int i = 0; i < nx; i++) {
         p[0][i] = p[1][i];
         p[ny - 1][i] = 0;
      }
    }

    // Copy u and v to un and vn
    for (int j = 0; j < ny; j++) {
      for (int i = 0; i < nx; i++) {
        un[j][i] = u[j][i];
        vn[j][i] = v[j][i];
      }
    }

    // Compute u and v
    for (int j = 1; j < ny - 1; j++) {
#define compute_uv(xlen)\
      do {\
        __m##xlen un_left = _mm##xlen##_loadu_ps(&un[j][i - 1]);\
        __m##xlen un_center = _mm##xlen##_loadu_ps(&un[j][i]);\
        __m##xlen un_right = _mm##xlen##_loadu_ps(&un[j][i + 1]);\
        __m##xlen un_top = _mm##xlen##_loadu_ps(&un[j - 1][i]);\
        __m##xlen un_bottom = _mm##xlen##_loadu_ps(&un[j + 1][i]);\
        \
        __m##xlen vn_left = _mm##xlen##_loadu_ps(&vn[j][i - 1]);\
        __m##xlen vn_center = _mm##xlen##_loadu_ps(&vn[j][i]);\
        __m##xlen vn_right = _mm##xlen##_loadu_ps(&vn[j][i + 1]);\
        __m##xlen vn_top = _mm##xlen##_loadu_ps(&vn[j - 1][i]);\
        __m##xlen vn_bottom = _mm##xlen##_loadu_ps(&vn[j + 1][i]);\
        \
        __m##xlen p_left = _mm##xlen##_loadu_ps(&p[j][i - 1]);\
        __m##xlen p_right = _mm##xlen##_loadu_ps(&p[j][i + 1]);\
        __m##xlen p_top = _mm##xlen##_loadu_ps(&p[j - 1][i]);\
        __m##xlen p_bottom = _mm##xlen##_loadu_ps(&p[j + 1][i]);\
        \
        /* Compute u*/\
        \
        /* u_term1 = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) */ \
        __m##xlen u_term1 = _mm##xlen##_sub_ps(un_center, _mm##xlen##_mul_ps(un_center, _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dt / dx), _mm##xlen##_sub_ps(un_center, un_left))));\
        \
        /* u_term2 = un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])*/\
        __m##xlen u_term2 = _mm##xlen##_mul_ps(un_center, _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dt / dy), _mm##xlen##_sub_ps(un_center, un_top)));\
        \
          /* u_term3 = dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1])*/\
        __m##xlen u_term3 = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dt / (2 * rho * dx)), _mm##xlen##_sub_ps(p_right, p_left));\
        \
        /* u_term4 = nu * dt / (dx * dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) */\
        /*         = c * (un_right + un_left - 2 * un_center) */ \
        /*         = c * (u_term4_a - u_term4_b) */\
        __m##xlen u_term4_a = _mm##xlen##_add_ps(un_right, un_left);\
        __m##xlen u_term4_b = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(2), un_center);\
        __m##xlen u_term4   = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(nu * dt / (dx * dx)), _mm##xlen##_sub_ps(u_term4_a, u_term4_b));\
        \
        /* u_term5 = nu * dt / (dy * dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]) */\
        /*         = c * (un_bottom + un_top - 2 * un_center)*/\
        /*         = c * (u_term5_a - u_term5_b)*/\
        /*         = c * (u_term5_a - u_term4_b) // take u_term5_b = u_term4_b to avoid redundant operation*/\
        __m##xlen u_term5_a = _mm##xlen##_add_ps(un_bottom, un_top);\
        __m##xlen u_term5   = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(nu * dt / (dy * dy)), _mm##xlen##_sub_ps(u_term5_a, u_term4_b));\
        \
        /* u_res = u_term1 - u_term2 - u_term3 + u_term4 + u_term5;*/\
        __m##xlen u_res = _mm##xlen##_sub_ps(u_term1, u_term2);\
        u_res = _mm##xlen##_sub_ps(u_res, u_term3);\
        u_res = _mm##xlen##_add_ps(u_res, u_term4);\
        u_res = _mm##xlen##_add_ps(u_res, u_term5);\
        \
        _mm##xlen##_storeu_ps(&u[j][i], u_res);\
        \
        /* Compute v */\
        \
        /* v_term1 = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])*/\
        __m##xlen v_term1 = _mm##xlen##_sub_ps(vn_center, _mm##xlen##_mul_ps(vn_center, _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dt / dx), _mm##xlen##_sub_ps(vn_center, vn_left))));\
        \
        /* v_term2 = vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) */\
        __m##xlen v_term2 = _mm##xlen##_mul_ps(vn_center, _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dt / dy), _mm##xlen##_sub_ps(vn_center, vn_top)));\
        \
          /* v_term3 = dt / (2 * rho * dy) * (p[j + 1][i] - p[j - 1][i])*/\
        __m##xlen v_term3 = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(dt / (2 * rho * dy)), _mm##xlen##_sub_ps(p_bottom, p_top));\
        \
        /* v_term4 = nu * dt / (dx * dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) */\
        /*         = c * (vn_right + vn_left - 2 * vn_center)*/\
        /*         = c * (v_term4_a - v_term4_b)*/\
        __m##xlen v_term4_a = _mm##xlen##_add_ps(vn_right, vn_left);\
        __m##xlen v_term4_b = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(2), vn_center);\
        __m##xlen v_term4   = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(nu * dt / (dx * dx)), _mm##xlen##_sub_ps(v_term4_a, v_term4_b));\
        \
        /* v_term5 = nu * dt / (dy * dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i])*/ \
        /*         = c * (vn_bottom + vn_top - 2 * vn_center)*/ \
        /*         = c * (v_term5_a - v_term5_b)*/ \
        /*         = c * (v_term5_a - v_term4_b) // take v_term5_b = v_term4_b to avoid redundant operation*/ \
        __m##xlen v_term5_a = _mm##xlen##_add_ps(vn_bottom, vn_top);\
        __m##xlen v_term5   = _mm##xlen##_mul_ps(_mm##xlen##_set1_ps(nu * dt / (dy * dy)), _mm##xlen##_sub_ps(v_term5_a, v_term4_b));\
        \
        /* v_res = v_term1 - v_term2 - v_term3 + v_term4 + v_term5;*/\
        __m##xlen v_res = _mm##xlen##_sub_ps(v_term1, v_term2);\
        v_res = _mm##xlen##_sub_ps(v_res, v_term3);\
        v_res = _mm##xlen##_add_ps(v_res, v_term4);\
        v_res = _mm##xlen##_add_ps(v_res, v_term5);\
        \
        _mm##xlen##_storeu_ps(&v[j][i], v_res);\
      }while(0)

      int i = 1;
      for (; i < nx - 17; i += 16) {
        compute_uv(512);
      }
      for (; i < nx - 9; i += 8) {
        compute_uv(256);
      }
#undef compute_uv

      // Handle remaining elements
      for (; i < nx - 1; i++) {
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                  un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                  dt / (2 * rho * dx) * (p[j][i + 1] - p[j][i - 1]) +
                  nu * dt / (dx * dx) * (un[j][i + 1] - 2 * un[j][i] + un[j][i - 1]) +
                  nu * dt / (dy * dy) * (un[j + 1][i] - 2 * un[j][i] + un[j - 1][i]);

        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                  vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                  dt / (2 * rho * dy) * (p[j + 1][i] - p[j - 1][i]) +
                  nu * dt / (dx * dx) * (vn[j][i + 1] - 2 * vn[j][i] + vn[j][i - 1]) +
                  nu * dt / (dy * dy) * (vn[j + 1][i] - 2 * vn[j][i] + vn[j - 1][i]);
      }
    }

    // Boundary conditions for u and v
    for (int j = 0; j < ny; j++) {
      u[j][0] = 0;
      u[j][nx - 1] = 0;
      v[j][0] = 0;
      v[j][nx - 1] = 0;
    }
    for (int i = 0; i < nx; i++) {
      u[0][i] = 0;
      u[ny - 1][i] = 1;
      v[0][i] = 0;
      v[ny - 1][i] = 0;
    }

    // This block serves for accuracy verification. You can undefine SIMD_ACCURACY_CHECK
#if defined(SIMD_ACCURACY_CHECK)
    {
      // With an alias trick, we can copy and reuse the code directly from 10_cavity.cpp.
      auto& u = r_u;
      auto& v = r_v;
      auto& p = r_p;
      auto& b = r_b;
      auto& un = r_un;
      auto& vn = r_vn;
      auto& pn = r_pn;

      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
          // Compute b[j][i]
          b[j][i] = rho * (1 / dt *
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2) -
                    2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * (v[j][i+1] - v[j][i-1]) / (2 * dx)) -
                    pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));

        }
      }

      for (int it=0; it<nit; it++) {

        // Copy p to pn
        for (int j=0; j<ny; j++) {
          for (int i=0; i<nx; i++) {
            pn[j][i] = p[j][i];
          }
        }

        for (int j=1; j<ny-1; j++) {
          for (int i=1; i<nx-1; i++) {
            // Compute p[j][i]
            p[j][i] = (pow(dy,2) * (pn[j][i+1] + pn[j][i-1]) +
                        pow(dx,2) * (pn[j+1][i] + pn[j-1][i]) -
                        b[j][i] * pow(dx,2) * pow(dy,2)) /
                        (2 * (pow(dx,2) + pow(dy,2)));
            }
        }

        for (int j=0; j<ny; j++) {
          // Compute p[j][0] and p[j][nx-1]
          p[j][0] = p[j][1];
          p[j][nx-1] = p[j][nx-2];
        }

        for (int i=0; i<nx; i++) {
          // Compute p[0][i] and p[ny-1][i]
          p[0][i] = p[1][i];
          p[ny-1][i] = 0;
        }
      }

      for (int j=0; j<ny; j++) {
        for (int i=0; i<nx; i++) {
          un[j][i] = u[j][i];
          vn[j][i] = v[j][i];
        }
      }

      for (int j=1; j<ny-1; j++) {
        for (int i=1; i<nx-1; i++) {
          // Compute u[j][i] and v[j][i]
          u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1]) -
                    un[j][i] * dt / dy * (un[j][i] - un[j - 1][i]) -
                    dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) +
                    nu * dt / (dx * dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) +
                    nu * dt / (dy * dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);

          v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1]) -
                    vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i]) -
                    dt / (2 * rho * dy) * (p[j+1][i] - p[j-1][i]) +
                    nu * dt / (dx * dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) +
                    nu * dt / (dy * dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
        }
      }

      for (int j=0; j<ny; j++) {
        // Compute u[j][0], u[j][nx-1], v[j][0], v[j][nx-1]
        u[j][0] = 0; u[j][nx-1] = 0;
        v[j][0] = 0; v[j][nx-1] = 0;
      }

      for (int i=0; i<nx; i++) {
        // Compute u[0][i], u[ny-1][i], v[0][i], v[ny-1][i]
        u[0][i] = 0; u[ny-1][i] = 1;
        v[0][i] = 0; v[ny-1][i] = 0;
      }
    }
    
#endif

    if (n % 10 == 0) {
      for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
          ufile << u[j][i] << " ";
          vfile << v[j][i] << " ";
          pfile << p[j][i] << " ";
        }
      }
      ufile << "\n";
      vfile << "\n";
      pfile << "\n";

#if defined(SIMD_ACCURACY_CHECK)
      auto u_norm = l2_norm(r_u, u, ny, nx);
      auto v_norm = l2_norm(r_v, v, ny, nx);
      auto p_norm = l2_norm(r_p, p, ny, nx);
      printf("n:%3d, ||r_u - u||:%g, ||r_v - v||:%g, ||r_p - p||:%g\n",
          n, u_norm, v_norm, p_norm);
#endif
    }
  }

  ufile.close();
  vfile.close();
  pfile.close();

  return 0;
}

