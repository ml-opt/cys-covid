#include <cassert>
#include <math.h>
#include <core/errors/mse.h>

namespace dnn_opt
{
namespace core
{
namespace errors
{

mse* mse::make()
{
  return new mse();
}

float mse::f(int size, int dim, const float* out, const float* exp) const
{
  float mse = 0.0f;

  for(int i = 0; i < size; i++)
  {
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      mse += pow(exp[p + j] - out[p + j], 2.0f);
    }
  }

  mse /= size * dim;

  return mse;
}

mse::mse()
{

}

} // namespace errors
} // namespace core
} // namespace dnn_opt
