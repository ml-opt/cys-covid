#include <core/generators/normal.h>

namespace dnn_opt
{
namespace core
{
namespace generators
{

normal* normal::make(float mean, float dev)
{
  return new normal(mean, dev);
}

void normal::generate(int count, float* params) const
{
  float ext = m_max - m_min;

  for(int i = 0; i < count; i++)
  {
    params[i] = (*m_distribution)(*m_generator) * ext + m_min;
  }
}

float normal::generate() const
{
  return (*m_distribution)(*m_generator);
}

void normal::set_constraints(int count, float* params) const
{
  float min = get_min();
  float max = get_max();

  for(int i = 0; i < count; i++)
  {
    params[i] = std::max(min, std::min(max, params[i]));
  }
}

normal::normal(float mean, float dev)
: generator(mean - dev, mean + dev)
{
  std::random_device device;

  m_generator = new std::mt19937(device());
  m_distribution = new std::normal_distribution<float>(mean, dev);
  m_mean = mean;
  m_dev = dev;
}

normal::~normal()
{
  delete m_generator;
  delete m_distribution;
}

} // namespace generators
} // namespace core
} // namespace dnn_opt
