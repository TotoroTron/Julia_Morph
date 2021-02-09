#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SFML/Graphics.hpp>

void julia(void);

void initColors(sf::Color* colors, int const max_iter);

__global__ void cudaJulia(int const h, sf::Color* d_counts, sf::Color* d_colors, int const max_iter,
    double const re_min, double const im_min, double const re_scale,
    double const im_scale, double P, double Q);