#include "algorithm"
#include "stdexcept"
#include <core/layers/max_pool.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

max_pool* max_pool::make(std::vector<int> in_shape,
                         std::vector<int> w_shape,
                         std::vector<int> s_shape)
{
  max_pool* result = new max_pool(in_shape, w_shape, s_shape);

  result->init();

  return result;
}

void max_pool::init()
{

}

void max_pool::prop(int size, const float* in, const float* params, float* out) const
{
  int out_dim = get_out_dim();

  int H = m_in_shape[0];
  int W = m_in_shape[1];
  int D = m_in_shape[2];

  int W_H = m_w_shape[0];
  int W_W = m_w_shape[1];

  int S_H = m_s_shape[0];
  int S_W = m_s_shape[1];

  for(int i = 0; i < size; i++)
  {
    for(int d = 0; d < D; d++)
    {
      for(int j = 0; j < out_dim; j++)
      {
        int out_idx = i * out_dim + j;

        int SH_H = j % (H / S_H) * S_H;                // shift height stride
        int SH_W = j % (W / S_W) * S_W;                // shift width stride

        int in_idx = H * W * D * i +                   // i-th example
                     H * W * d +                       // d-th channel
                     H * SH_H +                        // first height index
                     SH_W;                             // first width index

        out[out_idx] = in[in_idx];

        for(int h = 0; h < W_H; h++)
        {
          for(int w = 0; w < W_W; w++)
          {
            in_idx = H * W * D * i +                   // i-th example
                     H * W * d +                       // d-th channel
                     H * (SH_H + h)  +                 // current height index
                     SH_W + w ;                        // current width index

            out[out_idx] = std::max(out[out_idx], in[in_idx]);
          }
        }
      }
    }
  }
}

int max_pool::w_size() const
{
  return 0;
}

int max_pool::b_size() const
{
  return 0;
}

layer* max_pool::clone()
{
  return max_pool::make(m_in_shape, m_w_shape, m_s_shape);
}

max_pool::max_pool(std::vector<int> in_shape,
                   std::vector<int> w_shape,
                   std::vector<int> s_shape)
: layer(in_shape,
       {
          ((in_shape[0] - w_shape[0]) / s_shape[0] + 1),
          ((in_shape[1] - w_shape[1]) / s_shape[1] + 1) * in_shape[2]
       })
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
    throw std::out_of_range("wrong window shape, given dimensions: " +
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
    throw std::out_of_range("wrong stride shape, given dimensions: " +
                            std::to_string(s_shape.size()));
  }
}

} // namespace layers
} // namespace core
} // namespace dnn_opt
