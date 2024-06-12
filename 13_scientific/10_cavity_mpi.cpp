#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

// Uncomment the below line to verify accuracy.
//#define MPI_ACCURACY_CHECK

using namespace std;

#if defined(MPI_ACCURACY_CHECK)
typedef vector<vector<float>> matrix;
static float l2_norm(const matrix& v1, const vector<float>& v2, int ny, int nx);
#endif

int main(int argc, char **argv) {
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int rows_per_process = ny / size;
  const int remaining_rows = ny % size;

  // com_rows indicates the rows to compute in this process
  const int com_rows = rows_per_process + (rank < remaining_rows ? 1 : 0);

  // For the first process or the last one, pad 1 row. Otherwise, pad 2 rows.
  const int pad_rows = (rank == 0 || rank == size - 1)? 1 : 2;
  const int rows = com_rows + pad_rows;

#if 0 // Set it to 1 to print info.
  const int begin = rank * rows_per_process + min(rank, remaining_rows);
  const int end = begin + com_rows;
  printf("rank:%d, begin:%d, end:%d, rows:%d\n", rank, begin, end, rows);
#endif

  vector<float> u(rows * nx, 0.0);
  vector<float> v(rows * nx, 0.0);
  vector<float> p(rows * nx, 0.0);
  vector<float> b(rows * nx, 0.0);
  vector<float> pn(rows * nx, 0.0);
  vector<float> un(rows * nx, 0.0);
  vector<float> vn(rows * nx, 0.0);

#if defined(MPI_ACCURACY_CHECK)
  // Prefix r_ indicates reference.
  matrix r_u(ny,vector<float>(nx));
  matrix r_v(ny,vector<float>(nx));
  matrix r_p(ny,vector<float>(nx));
  matrix r_b(ny,vector<float>(nx));
  matrix r_un(ny,vector<float>(nx));
  matrix r_vn(ny,vector<float>(nx));
  matrix r_pn(ny,vector<float>(nx));
  for (int j=0; j<ny; j++) {
    for (int i=0; i<nx; i++) {
      r_u[j][i] = 0;
      r_v[j][i] = 0;
      r_p[j][i] = 0;
      r_b[j][i] = 0;
    }
  }
#endif

  // rows_info stores the number of rows handled by each process.
  // This info will be utilized when invoking MPI_Get/Put.
  vector<int> rows_info(size);
  for (int i = 0; i < size; i++) {
    const int com_rows = rows_per_process + (i < remaining_rows ? 1 : 0);
    const int pad_rows = (i == 0 || i == size - 1)? 1 : 2;
    rows_info[i] = com_rows + pad_rows;
  }

  // u_all, v_all and p_all are used to gather results from all processes.
  vector<float> u_all, v_all, p_all;
  vector<int> recvcounts(size), displs(size);
  if (rank == 0) {
    u_all.resize(ny * nx);
    v_all.resize(ny * nx);
    p_all.resize(ny * nx);

    int offset = 0;
    for (int i = 0; i < size; i++) {
      const int pad_rows = (i == 0 || i == size - 1)? 1 : 2;
      recvcounts[i] = (rows_info[i] - pad_rows) * nx;
      displs[i] = offset;
      offset += recvcounts[i];
    }
  }

  ofstream ufile, vfile, pfile;
  if (rank == 0) {
    ufile.open("u.dat");
    vfile.open("v.dat");
    pfile.open("p.dat");

#if defined(MPI_ACCURACY_CHECK)
    printf("Results by MPI will be verified.\n"
           "r_u, r_v and r_p are derived from CPU, whereas u, v and p are from MPI.\n"
           "||x|| represents l2 norm.\n");
#endif
  }

  MPI_Win win_u, win_v, win_p;
  MPI_Win_create(u.data(), rows * nx * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_u);
  MPI_Win_create(v.data(), rows * nx * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_v);
  MPI_Win_create(p.data(), rows * nx * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win_p);

  for (int n = 0; n < nt; n++) {

    for (int j = 1; j < rows - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
          // Compute b[j][i]
        b[j * nx + i] = rho * (1 / dt *
            ((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx) + (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) -
            pow((u[j * nx + i + 1] - u[j * nx + i - 1]) / (2 * dx), 2) -
            2 * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy) *
            (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2 * dx)) -
            pow((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy), 2));
      }
    }

    // p computation iteration
    for (int it = 0; it < nit; it++) {
    
       // Copy p to pn
      for (int j = 0; j < rows; j++) {
        for (int i = 0; i < nx; i++) {
          pn[j * nx + i] = p[j * nx + i];
        }
      }

      for (int j = 1; j < rows - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
          // Compute p[j][i]
          p[j * nx + i] = (pow(dy, 2) * (pn[j * nx + i + 1] + pn[j * nx + i - 1]) +
                           pow(dx, 2) * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
                           b[j * nx + i] * pow(dx, 2) * pow(dy, 2)) /
                          (2 * (pow(dx, 2) + pow(dy, 2)));
        }
      }

      // For the first process or the last one, pad 1 row. Otherwise, pad 2 rows.
      // So, j indexing must be cautious.
      for (int j = (rank == 0 ? 0 : 1); j < (rank == size - 1? rows : rows - 1); j++) {
        p[j * nx + 0] = p[j * nx + 1];
        p[j * nx + nx - 1] = p[j * nx + nx - 2];
      }

      if (rank == 0) {
        for (int i = 0; i < nx; i++) {
          p[0 * nx + i] = p[1 * nx + i];
        }
      }

      if (rank == size - 1) {
        for (int i = 0; i < nx; i++) {
          p[(rows - 1) * nx + i] = 0;
        }
      }

      // Communicate boundary rows for p.
      MPI_Win_fence(0, win_p);
      if (rank != 0) {
        const int pre_rank = rank - 1;
        const int pre_rank_rows = rows_info[pre_rank];
        MPI_Put(&p[1 * nx], nx, MPI_FLOAT, pre_rank, (pre_rank_rows-1) * nx, nx, MPI_FLOAT, win_p);
        MPI_Get(&p[0 * nx], nx, MPI_FLOAT, pre_rank, (pre_rank_rows-2) * nx, nx, MPI_FLOAT, win_p);
      }
      if (rank != size - 1) {
        const int nxt_rank = rank + 1;
        MPI_Put(&p[(rows - 2) * nx], nx, MPI_FLOAT, nxt_rank, 0 * nx, nx, MPI_FLOAT, win_p);
        MPI_Get(&p[(rows - 1) * nx], nx, MPI_FLOAT, nxt_rank, 1 * nx, nx, MPI_FLOAT, win_p);
      }
      MPI_Win_fence(0, win_p);
    }

    for (int j = 0; j < rows; j++) {
      for (int i = 0; i < nx; i++) {
        un[j * nx + i] = u[j * nx + i];
        vn[j * nx + i] = v[j * nx + i];
      }
    }

    for (int j = 1; j < rows - 1; j++) {
      for (int i = 1; i < nx - 1; i++) {
        // Compute u[j][i] and v[j][i]
        u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) -
                        un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                        dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) +
                        nu * dt / (dx * dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) +
                        nu * dt / (dy * dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);

        v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) -
                        vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
                        dt / (2 * rho * dy) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
                        nu * dt / (dx * dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) +
                        nu * dt / (dy * dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
      }
    }

    for (int j = (rank == 0 ? 0 : 1); j < ((rank == size - 1)? rows : rows - 1); j++) {
      u[j * nx + 0] = 0;
      u[j * nx + nx - 1] = 0;
      v[j * nx + 0] = 0;
      v[j * nx + nx - 1] = 0;
    }

    if (rank == 0) {
      for (int i = 0; i < nx; i++) {
        u[0 * nx + i] = 0;
        v[0 * nx + i] = 0;
      }
    }

    if (rank == size - 1) {
      for (int i = 0; i < nx; i++) {
        u[(rows - 1) * nx + i] = 1;
        v[(rows - 1) * nx + i] = 0;
      }
    }

    // Communicate boundary rows for u and v
    MPI_Win_fence(0, win_u);
    MPI_Win_fence(0, win_v);
    if (rank != 0) {
      const int pre_rank = rank - 1; // the previous neighbouring process ID.
      const int pre_rank_rows = rows_info[pre_rank];
      MPI_Put(&u[1 * nx], nx, MPI_FLOAT, pre_rank, (pre_rank_rows-1) * nx, nx, MPI_FLOAT, win_u);
      MPI_Get(&u[0 * nx], nx, MPI_FLOAT, pre_rank, (pre_rank_rows-2) * nx, nx, MPI_FLOAT, win_u);
      MPI_Put(&v[1 * nx], nx, MPI_FLOAT, pre_rank, (pre_rank_rows-1) * nx, nx, MPI_FLOAT, win_v);
      MPI_Get(&v[0 * nx], nx, MPI_FLOAT, pre_rank, (pre_rank_rows-2) * nx, nx, MPI_FLOAT, win_v);
    }
    if (rank != size - 1) {
      const int nxt_rank = rank + 1; // the next neighbouring process ID.
      const int nxt_rank_rows = rows_info[nxt_rank];
      MPI_Put(&u[(rows - 2) * nx], nx, MPI_FLOAT, nxt_rank, 0 * nx, nx, MPI_FLOAT, win_u);
      MPI_Get(&u[(rows - 1) * nx], nx, MPI_FLOAT, nxt_rank, 1 * nx, nx, MPI_FLOAT, win_u);
      MPI_Put(&v[(rows - 2) * nx], nx, MPI_FLOAT, nxt_rank, 0 * nx, nx, MPI_FLOAT, win_v);
      MPI_Get(&v[(rows - 1) * nx], nx, MPI_FLOAT, nxt_rank, 1 * nx, nx, MPI_FLOAT, win_v);
    }
    MPI_Win_fence(0, win_v);
    MPI_Win_fence(0, win_u);

#if defined(MPI_ACCURACY_CHECK)
    if (rank == 0) {
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
#endif /* End of MPI_ACCURACY_CHECK */

    if (n % 10 == 0) {
      // Gather data to root process before dumping to file
      const int begin = rank == 0 ? 0 : 1;
      const int r = com_rows;
      MPI_Gatherv(&u[begin * nx], r * nx, MPI_FLOAT, u_all.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gatherv(&v[begin * nx], r * nx, MPI_FLOAT, v_all.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Gatherv(&p[begin * nx], r * nx, MPI_FLOAT, p_all.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

      if (rank == 0) {
        for (int j = 0; j < ny; j++) {
          for (int i = 0; i < nx; i++) {
            ufile << u_all[j * nx + i] << " ";
            vfile << v_all[j * nx + i] << " ";
            pfile << p_all[j * nx + i] << " ";
          }
        }
        ufile << "\n";
        vfile << "\n";
        pfile << "\n";
#if defined(MPI_ACCURACY_CHECK)
      auto u_norm = l2_norm(r_u, u_all, ny, nx);
      auto v_norm = l2_norm(r_v, v_all, ny, nx);
      auto p_norm = l2_norm(r_p, p_all, ny, nx);
      printf("n:%3d, ||r_u - u||:%g, ||r_v - v||:%g, ||r_p - p||:%g\n",
          n, u_norm, v_norm, p_norm);
#endif
      }
    }
  }

  if (rank == 0) {
    ufile.close();
    vfile.close();
    pfile.close();
  }

  MPI_Win_free(&win_u);
  MPI_Win_free(&win_v);
  MPI_Win_free(&win_p);

  MPI_Finalize();
  return 0;
}

#if defined(MPI_ACCURACY_CHECK)
static float l2_norm(const matrix& v1, const vector<float>& v2, int ny, int nx)
{
  float sum = .0f;
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      float diff = v1[j][i] - v2[j * nx + i];
      sum += diff * diff;
    }
  }
  return sqrt(sum);
}
#endif
