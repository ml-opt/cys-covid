#include <core/activations/hard_limit.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

hard_limit* hard_limit::make()
{
  return new hard_limit();
}

void hard_limit::f(int size, int dim, const float* sum, float* out) const
{
  int n = size * dim;

  for(int i = 0; i < n; i++)
  {
    out[i] = sum[i] > 0 ? 1 : 0;
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
