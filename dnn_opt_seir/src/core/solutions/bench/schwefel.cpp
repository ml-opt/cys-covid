#include <math.h>
#include <core/solutions/bench/schwefel.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

schwefel* schwefel::make(generator* generator, unsigned int size)
{
  auto* result = new schwefel(generator, size);

  result->init();

  return result;
}

float schwefel::calculate_fitness()
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

schwefel::schwefel(generator* generator, unsigned int size )
: solution(generator, size)
{

}

schwefel::~schwefel()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
