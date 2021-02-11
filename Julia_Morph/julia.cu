#include <SFML/Graphics.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "julia.cuh"

void julia(void)
{   
    const int WIDTH = 1920; const int HEIGHT = 1080;
    const int HALF_WIDTH = WIDTH / 2;
    const int HALF_HEIGHT = HEIGHT / 2;
    const int AREA = HEIGHT * WIDTH;
    const float RATIO = (float) WIDTH / HEIGHT;

    const int max_iter = 1000;
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

    sf::Clock clock = sf::Clock::Clock(); sf::Time previousTime = clock.getElapsedTime(); sf::Time currentTime;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT, 32), "Julia Morph", sf::Style::Close | sf::Style::Titlebar);
    sf::Font font; font.loadFromFile("arial.ttf");
    sf::Text text; text.setFont(font); text.setCharacterSize(18); text.setFillColor(sf::Color::White);
    sf::Texture texture; sf::Sprite sprite; sf::Image image; image.create(HEIGHT, WIDTH);

    sf::Uint8* h_colorTable = new sf::Uint8[(max_iter + 1) * 4];
    sf::Uint8* d_colorTable;
    sf::Uint8* h_counts = new sf::Uint8[AREA * 4]; 
    sf::Uint8* d_counts;

    initColors(h_colorTable, max_iter);

    cudaMalloc(&d_counts, sizeof(sf::Uint8) * 4 * AREA);
    cudaMalloc(&d_colorTable, sizeof(sf::Uint8) * (max_iter + 1) * 4);
    cudaMemcpy(d_colorTable, h_colorTable, sizeof(sf::Uint8) * 4 * max_iter + 4, cudaMemcpyHostToDevice);

    

    int count1 = 0; int count2 = 0;

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
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R))
        {
            im_cent = 0.0; //imaginary axis center
            re_cent = 0.0; //real axis center
            zoom = 1.0; //inverse of zoom
            p = -1.26;
            q = 0.0;
        }
        int ff = 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) { ff = 100; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) { q += 0.00001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) { p -= 0.00001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) { q -= 0.00001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) { p += 0.00001 * ff; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Equal)) { zoom = zoom * 0.9; }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Hyphen)) { zoom = zoom * 1.1; }

        const dim3 blocksPerGrid(1080, 1, 1); const dim3 threadsPerBlock(960, 1, 1);
        cudaJulia<<<blocksPerGrid, threadsPerBlock>>>
            (WIDTH, HEIGHT, d_counts, d_colorTable, max_iter, re_min, im_min, re_scale, im_scale, p, q);
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

        char str[100];
        window.clear();
        texture.loadFromImage(image);
        sprite.setTexture(texture);
        window.draw(sprite);
        sprintf(str, "C = %1.5f + %1.5f*j\nZoom = %0.2E\nFPS = %3.0f", p, q, zoom, fps_t);
        text.setString(str);
        window.draw(text);
        window.display();
    }
    cudaFree(d_counts);
    cudaFree(d_colorTable);

    free(h_counts);
    free(h_colorTable);
    return;
}

void initColors(sf::Uint8* h_colorTable, int max_iter)
{
    float freq = 23.13 / max_iter;
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
    float P, float Q)
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
        while (iter < max_iter)
        {
            float tmp = (A * A) - (B * B) + P;
            B = (2 * A * B) + Q;
            A = tmp;
            if (A * A + B * B > 4)
                break;
            iter++;
        }
        int iter_4 = iter * 4;
        d_counts[m] = d_colorTable[iter_4];
        d_counts[m + 1] = d_colorTable[iter_4 + 1];
        d_counts[m + 2] = d_colorTable[iter_4 + 2];
        d_counts[m + 3] = d_colorTable[iter_4 + 3];
    }
    //julia set equation: z(n+1) = z(n)^2 + c
    //n = iteration
    //both z and c are complex numbers
    //z = A+jB
    //c = P+jQ
    //z(n+1) = (A^2 + B^2 + P) + j(2*A*B + Q)
};