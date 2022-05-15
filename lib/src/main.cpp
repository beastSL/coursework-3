#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "oracle.h"

namespace py = boost::python;
namespace np = boost::python::numpy;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(calc_predict_overloads, calc_predicts, 2, 3);

BOOST_PYTHON_MODULE(dynamic_regression_oracle)
{
    np::initialize();
    py::class_<DynamicRegressionOracle>("DynamicRegressionOracle", py::init<np::ndarray, py::object, py::object, int, py::object, bool>())
        .def("calc_predicts", &DynamicRegressionOracle::calc_predicts, calc_predict_overloads())
        .def("calc_grad_for_one_object", &DynamicRegressionOracle::calc_grad_for_one_object)
        .def("func", &DynamicRegressionOracle::func)
        .def("grad", &DynamicRegressionOracle::grad)
    ;
}