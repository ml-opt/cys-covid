#include <algorithm>
#include <core/layers/lc.h>
#include <iostream>
#include <cmath>
namespace dnn_opt
{
namespace core
{
namespace layers
{

lc* lc::make(std::vector<int> in_shape,
             std::vector<int> w_shape,
             std::vector<int> s_shape,
             int filters,
             const activation* activation)
{
  return new lc(in_shape, w_shape, s_shape, filters, activation);
}

void lc::init()
{

}

void lc::prop(int size, const float* in, const float* params, float* out) const
{
  int out_dim = get_out_dim();
  int weights = w_size();

  int H = m_in_shape[0];
  int W = m_in_shape[1];
  int D = m_in_shape[2];

  int W_H = m_w_shape[0];
  int W_W = m_w_shape[1];

  int S_H = m_s_shape[0];
  int S_W = m_s_shape[1];

  for(int i = 0; i < size; i++)
  {
    for(int j = 0; j < out_dim; j++)
    {
      int out_idx = i * out_dim + j;

      int SH_H = j % (H / S_H) * S_H;                 // shift height stride
      int SH_W = j % (W / S_W) * S_W;                 // shift width stride

      out[out_idx] = 0;

      for(int d = 0; d < D; d++)
      {
        for(int h = 0; h < W_H; h++)
        {
          for(int w = 0; w < W_W; w++)
          {
            int in_idx = H * W * D * i +               // i-th example
                         H * W * d +                   // d-th channel
                         H * (SH_H + h)  +             // h-th height index
                         SH_W + w;                     // w-th width index

            int ps_idx = W_H * W_W * D * j +           // j-th parameters
                         W_H * W_W * d +               // d-th depth index
                         W_H * h +                     // h-th height index
                         w;                            // w-th width index

            out[out_idx] += in[in_idx] * params[ps_idx];
          }
        }
      }
    }
  }

  /* sum bias param to the output of every kernel */

  for(int i = 0; i < size; i++)
  {
    for(int j = 0; j < out_dim; j++)
    {
      int out_idx = i * out_dim + j;

      out[out_idx] += params[weights + j];
    }
  }

  get_activation()->f(size, get_out_dim(), out, out);
}

int lc::w_size() const
{
  int D = m_in_shape[2];

  int W_H = m_w_shape[0];
  int W_W = m_w_shape[1];

  return get_out_dim() * W_H * W_W * D;
}

int lc::b_size() const
{
  return get_out_dim();
}

layer* lc::clone()
{
  return lc::make(get_in_shape(),
                  m_w_shape,
                  m_s_shape,
                  m_filters,
                  get_activation());
}

lc::lc(std::vector<int> in_shape,
       std::vector<int> w_shape,
       std::vector<int> s_shape,
       int filters,
       const activation* activation)
: layer(in_shape,
        {
          (in_shape[0] - w_shape[0]) / s_shape[0] + 1,
          (in_shape[1] - w_shape[1]) / s_shape[1] + 1,
           filters
        },
        activation)
{
  switch(w_shape.size())
  {
  case 1:
    m_w_shape = {in_shape[0], w_shape[0]};
    break;
  case 2:
    m_w_shape = w_shape;
    break;
  default:
    throw new std::out_of_range("wrong window shape, given dimensions: " +
                                std::to_string(w_shape.size()));
  }

  switch(s_shape.size())
  {
  case 1:
    m_s_shape = {1, s_shape[0]};
    break;
  case 2:
    m_s_shape = s_shape;
    break;
  default:
    throw new std::out_of_range("wrong stride shape, given dimensions: " +
                                std::to_string(s_shape.size()));
  }

  m_filters = filters;
}

} // namespace layers
} // namespace core
} // namespace dnn_opt
