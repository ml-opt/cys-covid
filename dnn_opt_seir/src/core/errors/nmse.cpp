#include <cassert>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <core/errors/nmse.h>

namespace dnn_opt
{
namespace core
{
namespace errors
{

nmse* nmse::make()
{
  return new nmse();
}

float nmse::f(int size, int dim, const float* out, const float* exp) const
{
  float mean = 0.0f;
  float var = 0.0f;
  float mse = 0.0f;
  float nmse = 0.0f;

  for(int i = 0; i < size; i++)
  {
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      mean += exp[p + j];
    }
  }

  mean /= size * dim;

  for(int i = 0; i < size; i++)
  {
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      var += pow(exp[p + j] - mean, 2.0f);
    }
  }

  var /= size * dim - 1;

  for(int i = 0; i < size; i++)
  {
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      mse += pow(exp[p + j] - out[p + j], 2.0f);
    }
  }

  nmse = mse / (size * dim * var);

  return nmse;
}

nmse::nmse()
{

}

} // namespace errors
} // namespace core
} // namespace dnn_opt
