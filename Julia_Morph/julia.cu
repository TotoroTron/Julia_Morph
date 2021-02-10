#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "julia.cuh"

void julia(void)
{   
    printf("%d", sizeof(sf::Color));
    const int WIDTH = 1000;
    const int HEIGHT = 700;
    const int HALF_WIDTH = WIDTH / 2;
    const int HALF_HEIGHT = HEIGHT / 2;
    const int AREA = HEIGHT * WIDTH;
    const double RATIO = (double)WIDTH / HEIGHT;

    int max_iter = 100;
    double im_cent = 0.0; //imaginary axis center
    double re_cent = 0.0; //real axis center
    double zoom = 1.0; //inverse of zoom]
    double p = -1.26;
    double q = 0.0;
    double re_min = re_cent - (zoom * RATIO);
    double re_max = re_cent + (zoom * RATIO);
    double im_min = im_cent - zoom;
    double im_max = im_cent + zoom;
    double re_scale = (re_max - re_min) / WIDTH;
    double im_scale = (im_max - im_min) / HEIGHT;

    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT, 32), "Julia Morph", sf::Style::Close | sf::Style::Resize);
    sf::Font font;
    font.loadFromFile("arial.ttf");
    
    sf::Text text;
    text.setFont(font);
    text.setCharacterSize(18);
    text.setFillColor(sf::Color::White);

    sf::Texture texture;
    sf::Sprite sprite;
    sf::Image image;
    image.create(HEIGHT, WIDTH);

    sf::Uint8* h_colorTable = new sf::Uint8[(max_iter + 1) * 4];
    sf::Uint8* d_colorTable = new sf::Uint8[(max_iter + 1) * 4];
    sf::Uint8* h_counts = new sf::Uint8[AREA * 4]; 
    sf::Uint8* d_counts = new sf::Uint8[AREA * 4];
    
    //h_counts = (sf::Uint8*) malloc(sizeof(sf::Uint8) * 4 * AREA);
    //h_colorTable = (sf::Uint8*) malloc(sizeof(sf::Uint8) * 4 * max_iter + 4);

    initColors(h_colorTable, max_iter);

    cudaMalloc(&d_counts, sizeof(sf::Uint8) * 4 * AREA);
    cudaMalloc(&d_colorTable, sizeof(sf::Uint8) * 4 * max_iter + 4);
    cudaMemcpy(d_colorTable, h_colorTable, sizeof(sf::Uint8) * 4 * max_iter + 4, cudaMemcpyHostToDevice);

    int count = 0;

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
        if (count == 50)
        {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
            {
                count = 0;
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                im_cent = im_cent - im_scale * (HALF_HEIGHT - (double)mousePos.y);
                re_cent = re_cent - re_scale * (HALF_WIDTH - (double)mousePos.x);
            }
        }
        else
        {
            count++;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) { window.close(); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R))
        {
            im_cent = 0.0; //imaginary axis center
            re_cent = 0.0; //real axis center
            zoom = 1.0; //inverse of zoom
            p = -1.26;
            q = 0.0;
        }
        int ff = 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) { ff = 1000; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) { q += 0.000001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) { p -= 0.000001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) { q -= 0.000001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) { p += 0.000001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Equal)) { zoom = zoom * 0.9; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Hyphen)) { zoom = zoom * 1.1; }

        cudaJulia <<<HEIGHT, WIDTH>>> (d_counts, d_colorTable, max_iter, re_min, im_min, re_scale, im_scale, p, q);
        cudaDeviceSynchronize();
        cudaMemcpy(h_counts, d_counts, sizeof(sf::Uint8) * 4 * AREA, cudaMemcpyDeviceToHost);

        image.create(WIDTH, HEIGHT, h_counts);

        for (int y = HALF_HEIGHT-10; y < HALF_HEIGHT+10; y++)
        {
            image.setPixel(HALF_WIDTH, y, sf::Color::White);
        }
        for (int x = HALF_WIDTH-10; x < HALF_WIDTH+10; x++)
        {
            image.setPixel(x, HALF_HEIGHT, sf::Color::White);
        }
        window.clear();
        texture.loadFromImage(image);
        sprite.setTexture(texture);
        window.draw(sprite);

        char str[100];
        sprintf(str, "C = %1.6f + %1.6f*j\nZoom = %0.2E", p, q, zoom);
        text.setString(str);
        window.draw(text);
        window.display();
        printf("_X");
    }
    cudaFree(d_counts);
    cudaFree(d_colorTable);

    free(h_counts);
    free(h_colorTable);
    return;
}

void initColors(sf::Uint8* h_colorTable, int const max_iter)
{
    double freq = 12.0 / max_iter;
    for (int i = 0; i < max_iter; i++) {
        h_colorTable[i * 4] = (uint8_t)(sin(freq * i + 3) * 127 + 128);
        h_colorTable[i * 4 + 1] = (uint8_t)(sin(freq * i + 5) * 127 + 128);
        h_colorTable[i * 4 + 2] = (uint8_t)(sin(freq * i + 1) * 127 + 128);
        h_colorTable[i * 4 + 3] = 255;
    }
    h_colorTable[max_iter * 4] = 0;
    h_colorTable[max_iter * 4 + 1] = 0;
    h_colorTable[max_iter * 4 + 2] = 0;
    h_colorTable[max_iter * 4 + 3] = 255;
}

__global__ void cudaJulia(sf::Uint8* d_counts, sf::Uint8* d_colorTable,
    int const max_iter, double const re_min, double const im_min, double const re_scale,
    double const im_scale, double P, double Q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;
    int y = blockIdx.x;

    //int idx = x * blockDim.x + y; //blockDim = width

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
    /*
    d_counts[idx * 4] = d_colorTable[iter * 4];
    d_counts[idx * 4 + 1] = d_colorTable[iter * 4 + 1];
    d_counts[idx * 4 + 2] = d_colorTable[iter * 4 + 2];
    d_counts[idx * 4 + 3] = d_colorTable[iter * 4 + 3];
    */
    d_counts[idx * 4] = d_colorTable[iter * 4];
    d_counts[idx * 4 + 1] = d_colorTable[iter * 4 + 1];
    d_counts[idx * 4 + 2] = d_colorTable[iter * 4 + 2];
    d_counts[idx * 4 + 3] = d_colorTable[iter * 4 + 3];
};