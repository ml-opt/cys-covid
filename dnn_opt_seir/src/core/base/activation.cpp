#include <core/base/activation.h>

namespace dnn_opt
{
namespace core
{

activation::numpy* activation::m_numpy = activation::numpy::make();

const activation* activation::get_numpy()
{
  return activation::m_numpy;
}

activation::~activation()
{

}

} // namespace core
} // namespace dnn_opt
