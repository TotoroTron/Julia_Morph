#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "julia.cuh"
#include <string>

void julia(void)
{   
    //const int WIDTH = 1920; const int HEIGHT = 1080;
    const int WIDTH = 2560; const int HEIGHT = 1440;
    const int HALF_WIDTH = WIDTH / 2;
    const int HALF_HEIGHT = HEIGHT / 2;
    const int AREA = HEIGHT * WIDTH;
    const float RATIO = (float) WIDTH / HEIGHT;

    int setType = 1;
    int max_iter = 200;
    float im_cent = 0.0; //imaginary axis center
    float re_cent = 0.0; //real axis center
    float zoom = 1.0; //inverse of zoom]
    float p = -1.26;
    float q = 0.0;
    float re_min = re_cent - (zoom * RATIO);
    float re_max = re_cent + (zoom * RATIO);
    float im_min = im_cent - zoom;
    float im_max = im_cent + zoom;
    float re_scale = (re_max - re_min) / WIDTH;
    float im_scale = (im_max - im_min) / HEIGHT;
    float fps;
    float fps_t = 0;
    int count1 = 0; int count2 = 0;

    sf::Clock clock = sf::Clock::Clock(); sf::Time previousTime = clock.getElapsedTime(); sf::Time currentTime;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT, 32), "Julia Morph", sf::Style::Close | sf::Style::Fullscreen);
    sf::Font font; font.loadFromFile("arial.ttf");
    sf::Text text; text.setFont(font); text.setCharacterSize(18); text.setFillColor(sf::Color::White);
    sf::Texture texture; sf::Sprite sprite; sf::Image image; image.create(HEIGHT, WIDTH);


    sf::Uint8* d_colorTable;
    sf::Uint8* h_counts = new sf::Uint8[AREA * 4]; 
    sf::Uint8* d_counts = new sf::Uint8[AREA * 4];

    cudaMalloc(&d_counts, sizeof(sf::Uint8) * 4 * AREA);


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
        if (count1 == 50)
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
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) { window.close(); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
        {
            im_cent = 0.0; //imaginary axis center
            re_cent = 0.0; //real axis center
            zoom = 1.0; //inverse of zoom
            p = -1.3;
            q = 0.0;
            max_iter = 240;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num1)) { setType = 1; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num2)) { setType = 2; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Num3)) { setType = 3; }
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
            sprintf(str2, "Experimental");
            break;
        }

        float ff = 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) { ff = 100; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LControl)) { ff = 0.001; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) { q += (0.00001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) { p -= (0.00001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) { q -= (0.00001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) { p += (0.00001 * ff); }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::E)) { zoom = zoom * 0.9; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q)) { zoom = zoom * 1.1; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R)) { if (max_iter < 10000) { max_iter = max_iter + ff; } }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F)) { if (max_iter > 100) { max_iter = max_iter - ff; } }

        sf::Uint8* h_colorTable = new sf::Uint8[(max_iter + 1) * 4];
        initColors(h_colorTable, max_iter);
        cudaMalloc(&d_colorTable, sizeof(sf::Uint8) * (max_iter + 1) * 4);
        cudaMemcpy(d_colorTable, h_colorTable, sizeof(sf::Uint8) * 4 * max_iter + 4, cudaMemcpyHostToDevice);

        const dim3 blocksPerGrid(1440, 1, 1); const dim3 threadsPerBlock(640, 1, 1);
        //const dim3 blocksPerGrid(1080 * 4, 1, 1); const dim3 threadsPerBlock(480, 1, 1);
        cudaJulia<<<blocksPerGrid, threadsPerBlock>>>
            (WIDTH, HEIGHT, d_counts, d_colorTable, max_iter, re_min, im_min, re_scale, im_scale, p, q, setType);
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

        currentTime = clock.getElapsedTime();
        fps = 1.0f / (currentTime.asSeconds() - previousTime.asSeconds()); // the asSeconds returns a float
        previousTime = currentTime;

        if (count2 == 50)
        {
            fps_t = fps;
            count2 = 0;
        }
        else
        {
            count2++;
        }

        char str1[200];
        window.clear();
        texture.loadFromImage(image);
        sprite.setTexture(texture);
        window.draw(sprite);
        sprintf(str1, "C = %1.5f + %1.5f*j\nZoom = %0.2E\nIterations = %i\nFPS = %3.0f\nSet Type = %s", p, q, zoom, max_iter, fps_t, str2);
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
    float freq = 60.3 / max_iter;
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
    float P, float Q, int setType)
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
                iter = experimental(iter, max_iter, A, B, P, Q);
                break;
        }
        
        int iter_4 = iter * 4;
        d_counts[m] = d_colorTable[iter_4];
        d_counts[m + 1] = d_colorTable[iter_4 + 1];
        d_counts[m + 2] = d_colorTable[iter_4 + 2];
        d_counts[m + 3] = d_colorTable[iter_4 + 3];
    }
};

__device__ int mandelbrot(int iter, int max_iter, float A, float B, float P, float Q)
{
    while (iter < max_iter)
    {
        float tmp = (A * A) - (B * B) + P;
        B = (2 * A * B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
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

__device__ int experimental(int iter, int max_iter, float A, float B, float P, float Q)
{
    while (iter < max_iter)
    {
        float tmp = (A*A*A - 3*A*B*B) + P;
        B = (-B*B*B+3*A*A*B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
            break;
        iter++;
    }
    return iter;
}