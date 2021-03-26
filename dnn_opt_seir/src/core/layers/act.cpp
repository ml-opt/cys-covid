#include <algorithm>
#include <core/layers/act.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

act* act::make(std::vector<int> in_shape, const activation* activation)
{
  act* result = new act(in_shape, activation);

  result->init();

  return result;
}

void act::init()
{

}

void act::prop(int size, const float* in, const float* params, float* out) const
{
  get_activation()->f(size, get_in_dim(), in, out);
}

int act::w_size() const
{
  return 0;
}

int act::b_size() const
{
  return 0;
}

layer* act::clone()
{
  return act::make(get_in_shape(), get_activation());
}

act::act(std::vector<int> in_shape, const activation* activation)
: layer(in_shape, in_shape, activation)
{

}

} // namespace layers
} // namespace core
} // namespace dnn_opt
