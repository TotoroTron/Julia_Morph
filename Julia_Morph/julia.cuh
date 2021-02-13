#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SFML/Graphics.hpp>

struct cuComplex {
    float r;
    float i;
    cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

void julia(void);

void initColors(sf::Uint8* h_colorTable, int max_iter);

__global__ void cudaJulia(int w, int h, sf::Uint8* d_counts, sf::Uint8* d_colorTable,
    int const max_iter, float re_min, float im_min, float re_scale, float im_scale,
    float P, float Q, int setType, float z, float x, float c);

__device__ int mandelbrot(int iter, int max_iter, float A, float B, float P, float Q);

__device__ int burningShip(int iter, int max_iter, float A, float B, float P, float Q);

__device__ int mandelCubed(int iter, int max_iter, float A, float B, float P, float Q);

__device__ int experimental(int iter, int max_iter, float A, float B, float P, float Q, float z, float x, float c);