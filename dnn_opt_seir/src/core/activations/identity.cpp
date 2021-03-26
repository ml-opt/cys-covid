#include <core/activations/identity.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

identity* identity::make()
{
  return new identity();
}

void identity::f(int size, int dim, const float* sum, float* out) const
{
  int n = size * dim;

  for(int i = 0; i < n; i++)
  {
    out[i] = sum[i];
  }
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
