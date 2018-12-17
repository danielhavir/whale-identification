#include <stdlib.h>
#include <iostream>
#include <torch/extension.h>

torch::Tensor generate_indices(torch::Tensor indices, torch::Tensor labels, float p, int64_t len)
{
  srand (time(NULL));
  torch::Tensor perm = torch::randperm(len);
  std::cout << len << std::endl;
  torch::Tensor pair_indices = torch::zeros({len, 2}, torch::dtype(torch::kInt64));
  
  for (int i = 0; i < len; i++)
  {
    torch::Tensor idx = indices[i];
    torch::Tensor rand_idx = indices[torch::randint(0, len, {1,}, torch::dtype(torch::kInt64))[0]];
    if (((float) rand() / (RAND_MAX)) < p)
    {
      rand_idx = idx;
    }
    else
    {
      while ((rand_idx == idx)[0])
      {
        rand_idx = indices[torch::randint(0, len, {1,}, torch::dtype(torch::kInt64))[0]];
      }
    }
    pair_indices[i][0] += idx;
    pair_indices[i][1] += rand_idx;
  }
  return pair_indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("generate_indices", &generate_indices, "Generate pairs of indices");;
}
