#include <math.h>
#include <core/solutions/bench/de_jung.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

de_jung* de_jung::make(generator *generator, unsigned int size)
{
  auto* result = new de_jung(generator, size);

  result->init();

  return result;
}

float de_jung::calculate_fitness()
{
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  for(int i = 0; i < size(); i++)
  {
    result += pow(params[i], 2.0f);
  }

  return result;
}

de_jung::de_jung(generator* generator, unsigned int size)
: solution(generator, size)
{

}

de_jung::~de_jung()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
