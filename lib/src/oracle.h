#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <optional>
#include <numeric>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>
#include <cctype>

namespace py = boost::python;
namespace np = boost::python::numpy;

namespace {
    class DynamicRegressionOracle {
public:
        DynamicRegressionOracle(const np::ndarray& fh,
                                const py::object& sp,
                                const py::object& learning_steps,
                                int64_t ar_depth,
                                const py::object& seas_depth,
                                bool fit_intercept) {
            this->fh = numpy_array_to_vector<int64_t>(fh);
            this->num_steps = *std::max_element(this->fh.begin(), this->fh.end());
            std::string learning_steps_classname = py::extract<std::string>(learning_steps.attr("__class__").attr("__name__"));
            if (learning_steps_classname == "str") {
                this->learning_steps = std::vector<int64_t>(this->num_steps);
                std::iota(this->learning_steps.begin(), this->learning_steps.end(), 1);
            } else {
                this->learning_steps = numpy_array_to_vector<int64_t>(np::from_object(learning_steps));
            }
            this->ar_depth = ar_depth;
            this->intercept_weight = (int64_t)fit_intercept;
            bool seasonal = !sp.is_none() && !seas_depth.is_none();
            if (seasonal) {
                this->sp = py::extract<int64_t>(sp);
                this->seas_depth = py::extract<int64_t>(seas_depth);
            } else {
                this->sp.reset();
                this->seas_depth.reset();
            }
            this->max_depth = this->ar_depth;
            this->ar_indices.resize(this->ar_depth);
            std::iota(this->ar_indices.begin(), this->ar_indices.end(), -this->ar_depth);
            this->relative_indices = this->ar_indices;
            if (this->sp.has_value()) {
                this->max_depth = std::max(this->ar_depth, this->sp.value() * this->seas_depth.value());
                std::vector<int64_t> full_seas_indices(this->seas_depth.value());
                for (int64_t i = 0; i < seas_depth; i++) {
                    full_seas_indices[i] = -this->sp.value() * this->seas_depth.value() + i * this->sp.value();
                }
                std::set_difference(
                    full_seas_indices.begin(),
                    full_seas_indices.end(),
                    this->ar_indices.begin(),
                    this->ar_indices.end(),
                    std::back_inserter(this->seas_indices)
                );
                this->relative_indices.resize(0);
                std::merge(
                    this->ar_indices.begin(),
                    this->ar_indices.end(),
                    this->seas_indices.begin(),
                    this->seas_indices.end(),
                    std::back_inserter(this->relative_indices)
                );
                std::reverse(this->seas_indices.begin(), this->seas_indices.end());
            }
            std::reverse(this->ar_indices.begin(), this->ar_indices.end());
            std::reverse(this->relative_indices.begin(), this->relative_indices.end());
        }

        np::ndarray calc_predicts(const np::ndarray& params,
                                const np::ndarray& object,
                                const py::object& fh = py::object()) const {
            std::vector<double> internal_params = numpy_array_to_vector<double>(params);
            // return vector_to_numpy_array(internal_params);
            std::vector<double> internal_object = numpy_array_to_vector<double>(object);
            std::vector<int64_t> internal_fh;
            if (fh.is_none()) {
                internal_fh = this->fh;
            } else {
                internal_fh = numpy_array_to_vector<int64_t>(np::from_object(fh));
            }
            std::vector<double> predicted_values;
            for (int64_t step = 0; step < this->num_steps; step++) {
                double intercept_part = internal_params[0] * this->intercept_weight;
                std::vector<int64_t> ar_indices = shrink_indices_to_size(this->ar_indices, internal_object.size());
                std::vector<double> ar_object = access_by_indices(internal_object, ar_indices);
                double ar_part = std::inner_product(ar_object.begin(), ar_object.end(), std::next(internal_params.begin()), (double)0);
                // return vector_to_numpy_array(std::vector<double>{ar_part});
                double cur_predict = intercept_part + ar_part;
                if (this->sp.has_value() && shrink_indices_to_size(this->seas_indices, internal_object.size()).size() > 0) {
                    std::vector<int64_t> seas_indices = shrink_indices_to_size(this->seas_indices, internal_object.size());
                    std::vector<double> seas_object = access_by_indices(internal_object, seas_indices);
                    double seas_part = std::inner_product(
                        seas_object.begin(),
                        seas_object.end(),
                        std::next(internal_params.begin(), 1 + this->ar_depth),
                        (double)0
                    );
                    cur_predict += seas_part;
                }
                predicted_values.push_back(cur_predict);
                internal_object.push_back(cur_predict);
            }
            return vector_to_numpy_array(access_by_indices(predicted_values, correct_fh(internal_fh)));
        }

        np::ndarray calc_grad_for_one_object(const np::ndarray& params,
                                           const np::ndarray& object,
                                           const np::ndarray& true_targets) const {
            std::vector<double> internal_params = numpy_array_to_vector<double>(params);
            std::vector<double> internal_object = numpy_array_to_vector<double>(object);
            std::vector<double> internal_true_targets = numpy_array_to_vector<double>(true_targets);
            std::vector<int64_t> internal_fh(this->num_steps);
            std::iota(internal_fh.begin(), internal_fh.end(), 1);
            std::vector<double> predicts = numpy_array_to_vector<double>(calc_predicts(params, object, vector_to_numpy_array(internal_fh)));
            std::vector<std::vector<double>> grads = {{}};
            grads[0].push_back(this->intercept_weight);
            std::vector<double> init_object = access_by_indices(internal_object, this->relative_indices);
            grads[0].insert(grads[0].end(), init_object.begin(), init_object.end());
            for (int64_t step = 1; step < num_steps; step++) {
                internal_object.push_back(predicts[step - 1]);
                std::vector<double> object_part;
                object_part.push_back(this->intercept_weight);
                std::vector<double> init_object = access_by_indices(internal_object, this->relative_indices);
                object_part.insert(object_part.end(), init_object.begin(), init_object.end());
                std::vector<int64_t> ar_indices = shrink_indices_to_size(this->ar_indices, step);
                std::vector<std::vector<double>> ar_object = access_by_indices(grads, ar_indices);
                std::vector<double> ar_grad_part = grads_params_product(ar_object, internal_params);
                std::vector<double> cur_grad(object_part.size());
                std::transform(object_part.begin(), object_part.end(), ar_grad_part.begin(), cur_grad.begin(), std::plus<double>());
                if (this->sp.has_value() && shrink_indices_to_size(this->seas_indices, step).size() > 0) {
                    std::vector<int64_t> seas_indices = shrink_indices_to_size(this->seas_indices, step);
                    std::vector<std::vector<double>> seas_object = access_by_indices(grads, seas_indices);
                    std::vector<double> seas_grad_part = grads_params_product(seas_object, internal_params, 1 + this->ar_depth);
                    std::transform(cur_grad.cbegin(), cur_grad.cend(), seas_grad_part.begin(), cur_grad.begin(), std::plus<double>());
                }
                grads.push_back(cur_grad);
            }
            predicts = access_by_indices(predicts, correct_fh(this->learning_steps));
            internal_true_targets = access_by_indices(internal_true_targets, correct_fh(this->learning_steps));
            grads = access_by_indices(grads, correct_fh(this->learning_steps));
            std::vector<double> error(this->learning_steps.size());
            std::transform(predicts.cbegin(), predicts.cend(), internal_true_targets.cbegin(), error.begin(), std::minus<double>());
            std::transform(error.cbegin(), error.cend(), error.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, 2));
            std::vector<double> output = grads_params_product(grads, error, 0);
            std::transform(
                output.cbegin(),
                output.cend(),
                output.begin(), 
                [size=this->learning_steps.size()](double x){ return x / size; }
            );
            return vector_to_numpy_array(output).copy();
        }

        double func(const np::ndarray& params, const np::ndarray& y) const {
            std::vector<double> internal_y = numpy_array_to_vector<double>(y);
            std::vector<double> errors;
            for (int64_t i = 0; i + this->max_depth + this->num_steps <= internal_y.size(); i++) {
                std::vector<double> object;
                std::copy(std::next(internal_y.begin(), i), std::next(internal_y.begin(), i + this->max_depth), std::back_inserter(object));
                std::vector<double> predicts = numpy_array_to_vector<double>(calc_predicts(params, vector_to_numpy_array(object)));
                std::vector<double> true_targets;
                std::copy(
                    std::next(internal_y.begin(), i + this->max_depth),
                    std::next(internal_y.begin(), i + this->max_depth + this->num_steps),
                    std::back_inserter(true_targets)
                );
                true_targets = access_by_indices(true_targets, correct_fh(this->fh));
                std::vector<double> point_errors;
                std::transform(predicts.begin(), predicts.end(), true_targets.begin(), std::back_inserter(point_errors), std::minus<double>());
                std::transform(point_errors.cbegin(), point_errors.cend(), point_errors.begin(), [](double x) {
                    return x * x;
                });
                errors.push_back(std::accumulate(point_errors.begin(), point_errors.end(), (double)0) / this->fh.size());
            }
            return std::accumulate(errors.begin(), errors.end(), (double)0) / errors.size();
        }

        np::ndarray grad(const np::ndarray& params, const np::ndarray& y) const {
            std::vector<double> internal_y = numpy_array_to_vector<double>(y);
            std::vector<double> internal_params = numpy_array_to_vector<double>(params);
            std::vector<std::vector<double>> grads;
            for (int64_t i = 0; i + this->max_depth + this->num_steps <= internal_y.size(); i++) {
                std::vector<double> object;
                std::copy(std::next(internal_y.begin(), i), std::next(internal_y.begin(), i + this->max_depth), std::back_inserter(object));
                std::vector<double> true_targets;
                std::copy(
                    std::next(internal_y.begin(), i + this->max_depth),
                    std::next(internal_y.begin(), i + this->max_depth + this->num_steps),
                    std::back_inserter(true_targets)
                );
                np::ndarray np_grad_for_one_object = calc_grad_for_one_object(
                    params,
                    vector_to_numpy_array(object),
                    vector_to_numpy_array(true_targets)
                );
                std::vector<double> grad_for_one_object = numpy_array_to_vector<double>(np_grad_for_one_object);
                grads.push_back(grad_for_one_object);
            }
            if (grads.size() == 0) {
                throw "max_depth is too big";
            }
            return vector_to_numpy_array(average_over_columns(grads)).copy();
        }

private:
        template <class T>
        std::vector<T> numpy_array_to_vector(const np::ndarray& data) const {
            int64_t input_size = data.shape(0);
            T* input_ptr = reinterpret_cast<T*>(data.get_data());
            std::vector<T> v(input_size);
            for (int64_t i = 0; i < input_size; ++i) {
                v[i] = *(input_ptr + i);
            }
            return v;
        }

        template <class T>
        np::ndarray vector_to_numpy_array(const std::vector<T>& data) const {
            int64_t v_size = data.size();
            py::tuple shape = py::make_tuple(v_size);
            py::tuple stride = py::make_tuple(sizeof(T));
            np::dtype dt = np::dtype::get_builtin<T>();
            np::ndarray output = np::from_data(&data[0], dt, shape, stride, py::object());
            return output.copy();
        }

        std::vector<int64_t> shrink_indices_to_size(const std::vector<int64_t>& indices, int64_t size) const {
            std::vector<int64_t> output;
            std::copy_if(indices.begin(), indices.end(), std::back_inserter(output), [size](int64_t index) {
                return index >= 0 && index < size || index < 0 && -index <= size;
            });
            return output;
        }

        std::vector<int64_t> correct_fh(const std::vector<int64_t>& fh) const {
            std::vector<int64_t> corrected_fh = fh;
            std::for_each(corrected_fh.begin(), corrected_fh.end(), [](int64_t& index) {
                --index;
            });
            return corrected_fh;
        }

        template <class T>
        std::vector<T> access_by_indices(const std::vector<T>& data, const std::vector<int64_t>& indices) const {
            std::vector<T> output;
            for (auto index : indices) {
                output.push_back(data[index >= 0 ? index : data.size() + index]);
            }
            return output;
        }

        std::vector<double> grads_params_product(const std::vector<std::vector<double>>& matrix,
                                                 const std::vector<double>& coefs,
                                                 int64_t params_offset = 1) const {
            return std::inner_product(
                matrix.begin(),
                matrix.end(),
                std::next(coefs.begin(), params_offset),
                std::vector<double>(matrix[0].size()), 
                [](const std::vector<double>& acc, const std::vector<double>& sum) {
                    std::vector<double> output(acc.size());
                    std::transform(acc.begin(), acc.end(), sum.begin(), output.begin(), std::plus<double>());
                    return output;
                }, [](const std::vector<double>& grad, double coef) {
                    std::vector<double> output(grad.size());
                    std::transform(grad.cbegin(), grad.cend(), output.begin(), std::bind(std::multiplies<double>(), std::placeholders::_1, coef));
                    return output;
                }
            );
        }

        std::vector<double> average_over_columns(const std::vector<std::vector<double>>& matrix) const {
            // return matrix[0];
            std::vector<double> output = std::accumulate(
                matrix.begin(),
                matrix.end(),
                std::vector<double>(matrix[0].size()),
                [](const std::vector<double>& a, const std::vector<double>& b) {
                    std::vector<double> output(a.size());
                    std::transform(a.cbegin(), a.cend(), b.cbegin(), output.begin(), std::plus<double>());
                    return output;
                });
            // return output;
            std::transform(output.cbegin(), output.cend(), output.begin(), [size=matrix.size()](double x) { return x / size; });
            return output;
        }

        std::vector<int64_t> fh;
        int64_t num_steps;
        std::vector<int64_t> learning_steps;
        int64_t ar_depth;
        int64_t intercept_weight;
        std::optional<int64_t> sp = 0;
        std::optional<int64_t> seas_depth = 0;
        int64_t max_depth;
        std::vector<int64_t> ar_indices;
        std::vector<int64_t> relative_indices;
        std::vector<int64_t> seas_indices;
    };
}