#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "julia.cuh"

void julia(void)
{   
    const int WIDTH = 1280;
    const int HEIGHT = 720;
    const int HALF_WIDTH = WIDTH / 2;
    const int HALF_HEIGHT = HEIGHT / 2;

    const int AREA = HEIGHT * WIDTH;
    const double RATIO = (double)WIDTH / HEIGHT;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT, 24), "Julia Morph", sf::Style::Close | sf::Style::Resize);

    sf::Texture texture;
    sf::Sprite sprite;
    sf::Image image;
    image.create(WIDTH, HEIGHT);

    int max_iter = 1000;
    double im_cent = 0.0; //imaginary axis center
    double re_cent = 0.0; //real axis center
    double zoom = 0.3; //inverse of zoom]
    double p = -1.1;
    double q = 0.2325;

    double re_min = re_cent - (zoom * RATIO);
    double re_max = re_cent + (zoom * RATIO);
    double im_min = im_cent - zoom;
    double im_max = im_cent + zoom;
    double re_scale = (re_max - re_min) / WIDTH;
    double im_scale = (im_max - im_min) / HEIGHT;

    sf::Color* colors;
    colors = (sf::Color*)malloc(sizeof(sf::Color) * max_iter + 4);
    initColors(colors, max_iter);
    sf::Color* h_counts;
    h_counts = (sf::Color*)malloc(sizeof(sf::Color) * AREA + 4); //host memory

    //device pointer
    sf::Color* d_counts;
    sf::Color* d_colors;

    //allocate device memory
    cudaMalloc(&d_counts, sizeof(int) * AREA);
    cudaMalloc(&d_colors, sizeof(sf::Color) * max_iter + 4);
    cudaMemcpy(d_colors, colors, sizeof(sf::Color) * max_iter + 4, cudaMemcpyHostToDevice);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            switch (event.type)
            {
            case sf::Event::Closed:
                window.close();
                break;
            }
        }

        re_min = re_cent - (zoom * RATIO);
        re_max = re_cent + (zoom * RATIO);
        im_min = im_cent - zoom;
        im_max = im_cent + zoom;
        re_scale = (re_max - re_min) / WIDTH;
        im_scale = (im_max - im_min) / HEIGHT;

        if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
        {
            sf::Vector2i mousePos = sf::Mouse::getPosition(window);
            im_cent = im_cent - im_scale * (HALF_HEIGHT - (double)mousePos.y);
            re_cent = re_cent - re_scale * (HALF_WIDTH - (double)mousePos.x);
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) { window.close(); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R))
        {
            im_cent = 0.0; //imaginary axis center
            re_cent = 0.0; //real axis center
            zoom = 0.3; //inverse of zoom]
            p = -1.1;
            q = 0.2325;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) { q += 0.0001; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) { p -= 0.0001; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) { q -= 0.0001; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) { p += 0.0001; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Hyphen)) { zoom = zoom * 0.9; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Equal)) { zoom = zoom * 1.1; }

        //kernel call <<< threadsPerBlock, numBlocks >>>
        cudaJulia <<<WIDTH, HEIGHT>>> (HEIGHT, d_counts, d_colors, max_iter, re_min, im_min, re_scale, im_scale, p, q);
        cudaDeviceSynchronize();
        //copy device data to host
        cudaMemcpy(h_counts, d_counts, sizeof(int) * AREA, cudaMemcpyDeviceToHost);

        for (int y = 0; y < HEIGHT; y++) //0 to 479
        {
            for (int x = 0; x < WIDTH; x++) //0 to 639
            {
                image.setPixel(x, y, h_counts[y + x * HEIGHT]);
            }
        }

        for (int y = 0; y < HEIGHT; y++)
        {
            image.setPixel(HALF_WIDTH, y, sf::Color::White);
        }
        for (int x = 0; x < WIDTH; x++)
        {
            image.setPixel(x, HALF_HEIGHT, sf::Color::White);
        }
        window.clear();
        texture.loadFromImage(image);
        sprite.setTexture(texture);
        window.draw(sprite);
        window.display();
    }
    cudaFree(d_counts);
    cudaFree(d_colors);

    free(h_counts);
    free(colors);
    return;
}

void initColors(sf::Color* colors, int const max_iter)
{
    double freq = 6.3 / max_iter;
    for (int i = 0; i < max_iter; i++) {
        uint8_t r = (uint8_t)(sin(freq * i + 3) * 127 + 128);
        uint8_t g = (uint8_t)(sin(freq * i + 5) * 127 + 128);
        uint8_t b = (uint8_t)(sin(freq * i + 1) * 127 + 128);
        colors[i] = { r,g,b };
    }
    colors[max_iter] = { 0,0,0 };
}

__global__ void cudaJulia(int const h, sf::Color* d_counts, sf::Color* d_colors,
    int const max_iter, double const re_min, double const im_min, double const re_scale,
    double const im_scale, double P, double Q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y = threadIdx.x;
    int x = blockIdx.x;

    //julia set equation: z(n+1) = z(n)^2 + c
    //n = iteration
    //both z and c are complex numbers
    //z = A+jB
    //c = P+jQ
    //z(n+1) = (A^2 + B^2 + P) + j(2*A*B + Q)
    
    double A = re_min + re_scale * x; //real
    double B = im_min + im_scale * y; //imag
    int iter = 0;
    
    while (iter < max_iter)
    {
        double tmp = (A * A) - (B * B) + P;
        B = (2 * A * B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
            break;
        iter++;
    }
    d_counts[idx] = d_colors[iter];
};