#include <cmath>
#include <core/activations/tan_h.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

tan_h* tan_h::make()
{
  return new tan_h();
}

void tan_h::f(int size, int dim, const float* sum, float* out) const
{
  int n = size * dim;

  for(int i = 0; i < n; i++)
  {
    out[i] = tanh(sum[i]);
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
