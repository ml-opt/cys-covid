#include <stdexcept>
#include <core/base/generator.h>

namespace dnn_opt
{
namespace core
{

generator::numpy* generator::m_numpy = generator::numpy::make();

const generator* generator::get_numpy()
{
  return generator::m_numpy;
}

float generator::get_min() const
{
  return m_min;
}

float generator::get_max() const
{
  return m_max;
}

void generator::set_min(float min)
{
  m_min = min;
}

void generator::set_max(float max)
{
  m_max = max;
}

float generator::get_ext() const
{
  return m_max - m_min;
}

generator::generator(float min, float max)
{
  if(min > max)
  {
    throw new std::range_error("generator min must be less than max");
  }

  m_min = min;
  m_max = max;
}

generator::~generator()
{

}

} // namespace core
} // namespace dnn_opt
