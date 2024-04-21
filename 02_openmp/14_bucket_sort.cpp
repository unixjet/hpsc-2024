#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  #pragma omp parallel
  {
    std::vector<int> local_bucket(range, 0);

    #pragma omp for nowait
    for (int i = 0; i < n; i++) {
      local_bucket[key[i]]++;
    }

    #pragma omp critical
    for (int i = 0; i < range; i++) {
      bucket[i] += local_bucket[i];
    }
  }

  std::vector<int> offset(range, 0);
  for (int i = 1; i < range; i++)
    offset[i] = offset[i-1] + bucket[i-1];

  std::vector<int> output(n);
  #pragma omp parallel for
  for (int i = 0; i < range; i++) {
    int j = offset[i];
    while (bucket[i] > 0) {
      output[j++] = i;
      bucket[i]--;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    key[i] = output[i];
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
