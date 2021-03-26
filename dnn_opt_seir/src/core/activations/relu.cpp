#include <algorithm>
#include <core/activations/relu.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

relu* relu::make()
{
  return new relu();
}

void relu::f(int size, int dim, const float* sum, float* out) const
{
  int n = size * dim;

  for(int i = 0; i < n; i++)
  {
    out[i] = std::max(0.0f, sum[i]);
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
