#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SFML/Graphics.hpp>

void julia(void);

void initColors(sf::Uint8* h_colorTable, int const max_iter);

__global__ void cudaJulia(sf::Uint8* d_counts, sf::Uint8* d_colorTable, int const max_iter,
    double const re_min, double const im_min, double const re_scale,
    double const im_scale, double P, double Q);