#include "algorithm"
#include <core/layers/fc.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

fc* fc::make(std::vector<int> in_shape,
             int neurons,
             const activation* activation)
{
  return new fc(in_shape, neurons, activation);
}

void fc::init()
{

}

void fc::prop(int size, const float* in, const float* params, float* out) const
{
  ws(size, in, params, out);
  get_activation()->f(size, get_out_dim(), out, out);
}

void fc::ws(int size, const float* in, const float* params, float* out) const
{
  int in_dim = get_in_dim();
  int out_dim = get_out_dim();
  int weight_size = w_size();

  std::fill_n(out, size * out_dim, 0.0f);

  /* transfer function */

  for(int i = 0; i < size; i++)
  {
    for(int k = 0; k < in_dim; k++)
    {
      for(int j = 0; j < out_dim; j++)
      {
        out[j * size + i] += in[k * size + i] * params[j * in_dim + k];
      }
    }
  }

  /* including bias terms into transfer function */

  for(int i = 0; i < out_dim; i++)
  {
    for(int j = 0; j < size; j++)
    {
      out[i * size + j] += params[weight_size + i];
    }
  }
}

int fc::w_size() const
{
  return get_in_dim() * get_out_dim();
}

int fc::b_size() const
{
  return get_out_dim();
}

layer* fc::clone()
{
  return fc::make(get_in_shape(), get_out_dim(), get_activation());
}

fc::fc(std::vector<int> in_shape, int neurons, const activation* activation)
  : layer(in_shape, {neurons}, activation)
{

}

} // namespace layers
} // namespace core
} // namespace dnn_opt
