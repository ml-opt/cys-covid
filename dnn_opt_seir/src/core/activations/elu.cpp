#include <cmath>
#include <core/activations/elu.h>

namespace dnn_opt
{
namespace core
{
namespace activations
{

elu* elu::make(float alpha)
{
  return new elu(alpha);
}

float elu::get_alpha()
{
  return _alpha;
}

void elu::f(int size, int dim, const float* sum, float* out) const
{
  int n = size * dim;

  for(int i = 0; i < n; i++)
  {
    if(sum[i] >= 0)
    {
      out[i] = sum[i];
    } else
    {
      out[i] = _alpha * (exp(out[i]) - 1);
    }
  }
}

elu::elu(float alpha)
{
  _alpha = alpha;
}

} // namespace activations
} // namespace core
} // namespace dnn_opt
