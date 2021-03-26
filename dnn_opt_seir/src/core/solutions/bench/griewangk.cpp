#include <math.h>
#include <core/solutions/bench/griewangk.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

griewangk* griewangk::make(generator* generator, unsigned int size)
{
  auto* result = new griewangk(generator, size);

  result->init();

  return result;
}

float griewangk::calculate_fitness()
{
  int n = size();
  float summatory = 0;
  float multiplier = 1;
  float result = 0;
  float* params = get_params();

  solution::calculate_fitness();

  for(int i = 0; i < n; i++)
  {
    summatory  += pow(params[i], 2.0) / 4000;
    multiplier *= cos(params[i] / sqrt(i + 1));
  }

  result = summatory - multiplier + 1;

  return result;
}

griewangk::griewangk(generator* generator, unsigned int size)
: solution(generator, size)
{

}

griewangk::~griewangk()
{

}

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt
