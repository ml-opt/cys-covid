#include <core/base/error.h>

namespace dnn_opt
{
namespace core
{

error::numpy* error::m_numpy = error::numpy::make();

const error* error::get_numpy()
{
  return error::m_numpy;
}

error::~error()
{

}

} // namespace core
} // namespace dnn_opt
