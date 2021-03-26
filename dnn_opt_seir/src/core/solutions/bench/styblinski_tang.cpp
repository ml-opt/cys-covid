#include <math.h>
#include <core/solutions/bench/styblinski_tang.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

styblinski_tang* styblinski_tang::make(generator* generator, unsigned int size)
{
  auto* result = new styblinski_tang(generator, size);

  result->init();

  return result;
}

float styblinski_tang::calculate_fitness()
{
  int n = size();
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    result += pow(params[i], 4) + 16 * pow(params[i], 2) + 5 * params[i];
  }

  return result / 2;
}

styblinski_tang::styblinski_tang(generator* generator, unsigned int size )
: solution(generator, size)
{

}

styblinski_tang::~styblinski_tang()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
