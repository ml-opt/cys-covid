#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace generators
{

uniform* uniform::make(float min, float max)
{
  return new uniform(min, max);
}

void uniform::generate(int count, float* params) const
{
  float ext = get_ext();

  for(int i = 0; i < count; i++)
  {
    params[i] = (*m_distribution)(*m_generator) * ext + m_min;
  }
}

float uniform::generate() const
{
  return (*m_distribution)(*m_generator);
}

void uniform::set_constraints(int count, float* params) const
{
  float min = get_min();
  float max = get_max();

  for(int i = 0; i < count; i++)
  {
    params[i] = std::max(min, std::min(max, params[i]));
  }
}

uniform::uniform(float min, float max) : generator(min, max)
{
  std::random_device device;

  m_generator = new std::mt19937(device());
  m_distribution = new std::uniform_real_distribution<>(0, 1);
}

uniform::~uniform()
{
  delete m_generator;
  delete m_distribution;
}

} // namespace generators
} // namespace core
} // namespace dnn_opt
