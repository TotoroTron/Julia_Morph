﻿#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "julia.cuh"
#include <string>

void julia(void)
{   
    //const int WIDTH = 640; const int HEIGHT = 360;
    //const int WIDTH = 1280; const int HEIGHT = 720;
    const int WIDTH = 1920; const int HEIGHT = 1080;
    //const int WIDTH = 2560; const int HEIGHT = 1440;
    
    //const dim3 blocksPerGrid(640, 1, 1); const dim3 threadsPerBlock(256, 1, 1);
    //const dim3 blocksPerGrid(320, 1, 1); const dim3 threadsPerBlock(720, 1, 1);
    const dim3 blocksPerGrid(160, 1, 1); const dim3 threadsPerBlock(240, 1, 1);
    //const dim3 blocksPerGrid(1440 * 2, 1, 1); const dim3 threadsPerBlock(640, 1, 1);

    const int HALF_WIDTH = WIDTH / 2;
    const int HALF_HEIGHT = HEIGHT / 2;
    const int AREA = HEIGHT * WIDTH;
    const float RATIO = (float) WIDTH / HEIGHT;

    int setType = 1;
    int max_iter = 200;
    float im_cent = 0.0; //imaginary axis center
    float re_cent = 0.0; //real axis center
    float zoom = 1.0;
    float p = -1.5; //C imaginary component
    float q = -0.1; //C real componenet
    
    float s = -3.75; //mandelbox s dimention
    float r = 1.65; //mandelbox r dimension
    float f = 1.0; //mandelbox f dimension
    float re_min = re_cent - (zoom * RATIO);
    float re_max = re_cent + (zoom * RATIO);
    float im_min = im_cent - zoom;
    float im_max = im_cent + zoom;
    float re_scale = (re_max - re_min) / WIDTH;
    float im_scale = (im_max - im_min) / HEIGHT;
    float fps;
    float fps_t = 0;
    int count1 = 0; int count2 = 0;
    float ff = 1;

    sf::Clock clock = sf::Clock::Clock(); sf::Time previousTime = clock.getElapsedTime(); sf::Time currentTime;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Julia Morph", sf::Style::Close | sf::Style::Fullscreen);
    sf::Font font; font.loadFromFile("arial.ttf");
    sf::Text text; text.setFont(font); text.setCharacterSize(16); text.setFillColor(sf::Color::White);
    sf::Texture texture;
    texture.create(WIDTH, HEIGHT);
    sf::Sprite sprite;
    sf::Uint8* d_colorTable;
    sf::Uint8* h_counts = new sf::Uint8[AREA * 4]; 
    sf::Uint8* d_counts = new sf::Uint8[AREA * 4];
    cudaMalloc(&d_counts, sizeof(sf::Uint8) * 4 * AREA);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
            if( event.type == sf::Event::Closed)
                window.close();
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) { window.close(); }
        if (count1 == 10)
        {
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left))
            {
                count1 = 0;
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                im_cent = im_cent - im_scale * (HALF_HEIGHT - (float)mousePos.y);
                re_cent = re_cent - re_scale * (HALF_WIDTH - (float)mousePos.x);
            }
        }
        else
        {
            count1++;
        }
        float ff = 1.0; float nn = 1.0;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
        {
            im_cent = 0.0; re_cent = 0.0; zoom = 1.0; p = -1.5; q = -0.1; max_iter = 200; 
            s = -3.75; r = 1.65; f = 1.0;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num1)) { setType = 1; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num2)) { setType = 2; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num3)) { setType = 3; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num4)) { setType = 4; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) { ff = 10; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) { ff = 0.01; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LAlt)) { nn = -1.0; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) { q = q + (0.0001 * ff); s = s + (0.001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) { p = p - (0.0001 * ff); r = r - (0.001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) { q = q - (0.0001 * ff); s = s - (0.001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) { p = p + (0.0001 * ff); r = r + (0.001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Z)) { f = f - (0.001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::C)) { f = f + (0.001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::E)) { zoom = zoom * 0.95; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q)) { zoom = zoom * 1.05; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R)) { if (max_iter < 10000) { max_iter = max_iter + ff; } }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F)) { if (max_iter > 20) { max_iter = max_iter - ff; } }

        re_min = re_cent - (zoom * RATIO);
        re_max = re_cent + (zoom * RATIO);
        im_min = im_cent - zoom;
        im_max = im_cent + zoom;
        re_scale = (re_max - re_min) / WIDTH;
        im_scale = (im_max - im_min) / HEIGHT;
        
        sf::Uint8* h_colorTable = new sf::Uint8[(max_iter + 1) * 4];
        initColors(h_colorTable, max_iter);
        cudaMalloc(&d_colorTable, sizeof(sf::Uint8) * (max_iter + 1) * 4);
        cudaMemcpy(d_colorTable, h_colorTable, sizeof(sf::Uint8) * (max_iter + 1) * 4, cudaMemcpyHostToDevice);

        cudaJulia<<<blocksPerGrid, threadsPerBlock>>>
            (WIDTH, HEIGHT, d_counts, d_colorTable, max_iter, re_min, im_min, re_scale, im_scale, p, q, setType, s, r, f);
        cudaDeviceSynchronize();
        cudaMemcpy(h_counts, d_counts, sizeof(sf::Uint8) * 4 * AREA, cudaMemcpyDeviceToHost);

        currentTime = clock.getElapsedTime();
        fps = 1.0f / (currentTime.asSeconds() - previousTime.asSeconds());
        previousTime = currentTime;
        if (count2 == 20) { fps_t = fps; count2 = 0; } else { count2++; }
        
        char str1[200];
        char str2[100];
        switch (setType)
        {
        case 1:
            sprintf(str2, "Burning Ship");
            break;
        case 2:
            sprintf(str2, "Mandelbrot");
            break;
        case 3:
            sprintf(str2, "Cubic Mandelbrot");
            break;
        case 4:
            sprintf(str2, "Experimental");
            break;
        }
        sprintf(str1, "C = %1.5f + %1.5f*j\nZoom = %0.2E\nIterations = %i\nFPS = %3.0f\nSet Type = %s\ns = %1.2f, r = %1.2f, f = %1.2f", p, q, zoom, max_iter, fps_t, str2, s, r, f);
        window.clear();
        texture.update(h_counts);
        sprite.setTexture(texture);
        window.draw(sprite);
        text.setString(str1);
        window.draw(text);
        window.display();
        cudaFree(d_colorTable);
        free(h_colorTable);
    }
    cudaFree(d_counts);
    free(h_counts);
    
    return;
}

void initColors(sf::Uint8* h_colorTable, int max_iter)
{
    float freq = 6.3 / max_iter;
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

__global__ void cudaJulia(int w, int h, sf::Uint8* d_counts, sf::Uint8* d_colorTable,
    int const max_iter, float re_min, float im_min, float re_scale, float im_scale,
    float P, float Q, int setType, float Z, float X, float C)
{   
    int pixPerThread = w * h / (gridDim.x * blockDim.x);
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int offset = pixPerThread * tid;

    for (int i = offset; i < offset + pixPerThread; i++)
    {
        int x = i % w;
        int y = i / w;
        int m = 4* (x + y * w);
        float A = re_min + re_scale * x;
        float B = im_min + im_scale * y;
        int iter = 0;

        switch (setType)
        {
            case 1:
                iter = burningShip(iter, max_iter, A, B, P, Q);
                break;
            case 2:
                iter = mandelbrot(iter, max_iter, A, B, P, Q);
                break;
            case 3:
                iter = mandelCubed(iter, max_iter, A, B, P, Q);
                break;
            case 4:
                iter = experimental(iter, max_iter, A, B, Z, X, C);
                break;
        }
        
        int iter_4 = iter * 4;
        
        d_counts[m] = d_colorTable[iter_4];
        d_counts[m + 1] = d_colorTable[iter_4 + 1];
        d_counts[m + 2] = d_colorTable[iter_4 + 2];
        d_counts[m + 3] = 255;
    }
};

__device__ int mandelbrot(int iter, int max_iter, float A, float B, float P, float Q)
{
    while (iter < max_iter)
    {
        float tmp = (A*A) - (B*B) + P;
        B = (2 * A * B) + Q;
        A = tmp;
        if (A*A + B*B > 4)
            break;
        iter++;
    }
    return iter;
}

__device__ int burningShip(int iter, int max_iter, float A, float B, float P, float Q)
{
    while (iter < max_iter)
    {
        float tmp = (A * A) - (B * B) + P;
        B = fabsf(2 * A * B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
            break;
        iter++;
    }
    return iter;
}

__device__ int mandelCubed(int iter, int max_iter, float A, float B, float P, float Q)
{
    while (iter < max_iter)
    {
        float tmp = (A * A * A - 3 * A * B * B) + P;
        B = (-B * B * B + 3 * A * A * B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
            break;
        iter++;
    }
    return iter;
}

__device__ int experimental(int iter, int max_iter, float A, float B, float s, float r, float f)
{   //Mandelbox
    float r_sq = r * r;
    float X = 0.0; float Y = 0.0;
    while (iter < max_iter)
    {
        float mag_sq = (X * X) + (Y * Y);
        float mag = sqrtf(mag_sq);
        if (mag > 4.0)
            break;
        iter++;

        if (X > 1.0)
            X = 2.0 - X;
        else if (X < -1.0)
            X = -2.0 - X;

        if (Y > 1.0)
            Y = 2.0 - Y;
        else if (Y < -1.0)
            Y = -2.0 - Y;

        X = X * f;
        Y = Y * f;

        if (mag < r_sq)
        {
            X = X / r_sq;
            Y = Y / r_sq;
        }
        else if (mag_sq < 1.0)
        {
            X = X / mag;
            Y = Y / mag;
        }

        X = X * s + A;
        Y = Y * s + B;

    }
    return iter;
}