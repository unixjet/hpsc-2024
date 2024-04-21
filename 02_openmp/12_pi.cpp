#include <cstdio>

int main() {
  int n = 100000;
  double dx = 1. / n;
  double pi = 0;
  double x;

#pragma omp parallel for private(x) reduction(+:pi)
  for (int i=0; i<n; i++) {
    x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  printf("%17.15f\n",pi);
}
