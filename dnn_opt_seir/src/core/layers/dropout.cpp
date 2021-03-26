#include <algorithm>
#include <core/layers/dropout.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

dropout* dropout::make(std::vector<int> in_shape, float p)
{
  dropout* result = new dropout(in_shape, p);

  result->init();

  return result;
}

void dropout::init()
{
  m_r = new float[get_out_dim()];
  m_generator = generators::uniform::make(0.0f, 1.0f);
}

void dropout::prop(int size, const float* in, const float* params, float* out) const
{
  int dim = get_out_dim();

  m_generator->generate(dim, m_r);

  for(int i = 0; i < size; i++)
  {
    for(int j = 0; j < dim; j++)
    {
      if(m_r[j] < m_p)
      {
        out[i * dim + j] = 0.0f;
      } else
      {
        out[i * dim + j] = in[i * dim + j];
      }
    }
  }
}

int dropout::w_size() const
{
  return 0;
}

int dropout::b_size() const
{
  return 0;
}

layer* dropout::clone()
{
  return dropout::make(get_in_shape(), m_p);
}

dropout::dropout(std::vector<int> in_shape, float p)
: layer(in_shape, in_shape)
{
  m_p = p;
}

dropout::~dropout()
{
  delete[] m_r;
  delete m_generator;
}

} // namespace layers
} // namespace core
} // namespace dnn_opt
