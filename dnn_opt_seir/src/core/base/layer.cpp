#include "functional"
#include "algorithm"
#include "numeric"
#include "stdexcept"
#include <core/base/layer.h>

namespace dnn_opt
{
namespace core
{

int layer::size() const
{
  return w_size() + b_size();
}

int layer::get_in_dim() const
{
  return std::accumulate(m_in_shape.begin(),
                         m_in_shape.end(),
                         1.0f,
                         std::multiplies<int>());
}

int layer::get_out_dim() const
{
  return std::accumulate(m_out_shape.begin(),
                         m_out_shape.end(),
                         1.0f,
                         std::multiplies<int>());
}

std::vector<int> layer::get_in_shape() const
{
  return m_in_shape;
}

std::vector<int> layer::get_out_shape() const
{
  return m_out_shape;
}

const activation* layer::get_activation() const
{
  return m_activation;
}

layer::layer(std::vector<int> in_shape,
             std::vector<int> out_shape,
             const activation* activation)
: m_activation(activation)
{
  switch(in_shape.size())
  {
  case 1:
    m_in_shape = {1, in_shape[0], 1};
    break;
  case 2:
    m_in_shape = {in_shape[0], in_shape[1], 1};
    break;
  case 3:
    m_in_shape = in_shape;
    break;
  default:
    throw new std::out_of_range("wrong input shape, dimensions, given: " +
                                std::to_string(in_shape.size()));
  }

  switch(out_shape.size())
  {
  case 1:
    m_out_shape = {1, out_shape[0], 1};
    break;
  case 2:
    m_out_shape = {out_shape[0], out_shape[1], 1};
    break;
  case 3:
    m_out_shape = out_shape;
    break;
  default:
    throw new std::out_of_range("wrong input shape, given dimensions: " +
                                std::to_string(out_shape.size()));
  }
}

layer::~layer()
{

}

} // namespace core
} // namespace dnn_opt
