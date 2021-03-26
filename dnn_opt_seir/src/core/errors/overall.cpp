#include <math.h>
#include <core/errors/overall.h>

namespace dnn_opt
{
namespace core
{
namespace errors
{

overall* overall::make()
{
  return new overall();
}

float overall::f(int size, int dim, const float* out, const float* exp) const
{
  float result = 0.0f;

  for(int i = 0; i < size; i++)
  {
    int p = i * dim;

    for(int j = 0; j < dim; j++)
    {
      if(exp[p + j] != out[p + j])
      {
        result++;
        break;
      }
    }
  }

  result /= size;

  return result;
}

overall::overall()
{

}

} // namespace errors
} // namespace core
} // namespace dnn_opt
