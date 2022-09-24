#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <assert.h>
#include <stdio.h>

namespace py = pybind11;

void matrix_multiply(const float *A, const float *B, float *C, size_t a_row, size_t a_col, size_t b_row, size_t b_col, size_t start_row)
{
    assert (a_col == b_row);
    
    for(size_t i = 0; i < a_row; i++) {
        for(size_t j =0; j < b_col; j++) {
            C[i * b_col + j] = 0;
            for(size_t k = 0; k < a_col; k++) {
                C[i * b_col + j] += A[i*a_col + k + (start_row * a_col)] * B[k*b_col + j];
            }
        }
    }
}

void matrix_multiply_T(const float *A, const float *B, float *C, size_t a_row, size_t a_col, size_t b_row, size_t b_col, size_t start_row)
{
    assert (a_row == b_row);
    for(size_t i=0; i < a_col; i++) {
        for(size_t j=0; j < b_col; j++) {
            C[i*b_col+j] = 0;
            for(size_t k=0; k<b_row; k++) {
                C[i*b_col+j] += A[(start_row*a_col) + k*a_col + i] * B[k*b_col + j];
            }
        }
    }
}
void normalize(float *A, size_t n_row, size_t n_col) {
    float *den = new float[n_row];
    
    for(size_t i = 0; i < n_row; i++) {
        den[i] = 0;
        for(size_t j = 0; j < n_col; j++) {
            A[i * n_col + j] = exp(A[i*n_col +j]);
            den[i] += A[i * n_col + j]; 
        }
    }
    for (size_t i = 0; i < n_row; i++) {
        for(size_t j = 0; j <n_col; j++) {
            A[i * n_col +j] = A[i * n_col + j] / den[i];
        }
    }
}

void compute_diff(float *theta_x, size_t n_row, size_t n_col, const unsigned char *y, size_t start) {
    for (size_t i = 0; i < n_row; i++) {
        theta_x[i * n_col + y[start + i]] -= 1;
    }
}

void sgd(float *theta, float *grad, float lr, size_t n) {
    for(size_t i = 0; i < n; i++) {
        theta[i] = theta[i] - lr * grad[i];
    }
}

void print_matrix(const float *X, size_t n_row, size_t n_col) {
    for(size_t i = 0; i < n_row; i++) {
        std::cout<<"\n";
        for(size_t j =0; j < n_col; j++) {
            std::cout << " " << X[i*n_col +j];
        }
    }
    std::cout<<"\n";
    
}


void testing_mm() {
    float *A = new float[18];
    float *B = new float[6];
    float *C = new float[4];
    for(int i =0; i< 18; i++) A[i] = i;
    for(int j =0; j< 6; j++) B[j] = j;
    print_matrix(A, 6, 3);
    print_matrix(B, 3, 2);
    matrix_multiply(A, B, C, 2, 3, 3, 2, 4);
    print_matrix(C, 2, 2);
    
}


void testing_a_t_mm() {
    float *A = new float[12];
    float *B = new float[6];
    float *C = new float[4];
    for(int i =0; i< 12; i++) A[i] = i;
    for(int j =0; j< 6; j++) B[j] = j;
    print_matrix(A, 6, 2);
    print_matrix(B, 3, 2);
    matrix_multiply_T(A, B, C, 3, 2, 3, 2, 3);
    print_matrix(C, 2, 2);
    
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    
    size_t batch_start = 0;
    while (batch_start < m) {
        float *theta_x = new float[batch*k];
        float *grad = new float[n*k];
        matrix_multiply(X, theta, theta_x, batch, n, n, k, batch_start);
        normalize(theta_x, batch, k);
        compute_diff(theta_x, batch, k, y, batch_start);
        matrix_multiply_T(X, theta_x, grad, batch, n, batch, k, batch_start);
        for(size_t i =0; i < n*k; i++) {grad[i] = 1.0/(float)batch * grad[i];}
        sgd(theta, grad, lr, n*k);
        batch_start += batch;
    }
    /// END YOUR CODE
}






/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
