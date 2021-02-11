#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <SFML/Graphics.hpp>

void julia(void);

void initColors(sf::Uint8* h_colorTable, int max_iter);

__global__ void cudaJulia(int w, int h, sf::Uint8* d_counts, sf::Uint8* d_colorTable,
    int const max_iter, float re_min, float im_min, float re_scale, float im_scale,
    float P, float Q);