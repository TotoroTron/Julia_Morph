#include <SFML/Graphics.hpp>
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
    const double RATIO = (double) WIDTH / HEIGHT;

    int setType = 4;
    int max_iter = 20;
    double im_cent = 0.0; //imaginary axis center
    double re_cent = 0.0; //real axis center
    double zoom = 1.0;
    double p = -1.5; //C imaginary component
    double q = -0.1; //C real componenet
    
    double s = -2.75; //mandelbox s dimention
    double r = 0.75; //mandelbox r dimension
    double f = 1.0; //mandelbox f dimension
    double re_min = re_cent - (zoom * RATIO);
    double re_max = re_cent + (zoom * RATIO);
    double im_min = im_cent - zoom;
    double im_max = im_cent + zoom;
    double re_scale = (re_max - re_min) / WIDTH;
    double im_scale = (im_max - im_min) / HEIGHT;
    double fps;
    double fps_t = 0;
    int count1 = 0; int count2 = 0;
    double ff = 1;

    sf::Clock clock = sf::Clock::Clock(); sf::Time previousTime = clock.getElapsedTime(); sf::Time currentTime;
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGHT), "Julia Morph", sf::Style::Close | sf::Style::Titlebar);
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
                im_cent = im_cent - im_scale * (HALF_HEIGHT - (double)mousePos.y);
                re_cent = re_cent - re_scale * (HALF_WIDTH - (double)mousePos.x);
            }
        }
        else
        {
            count1++;
        }
        double ff = 1.0; double nn = 1.0;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
        {
            im_cent = 0.0; re_cent = 0.0; zoom = 1.0; p = -1.5; q = -0.1; max_iter = 20; 
            s = -2.75; r = 0.75; f = 1.0;
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
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::F)) { if (max_iter > 10) { max_iter = max_iter - ff; } }

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
    int const max_iter, double re_min, double im_min, double re_scale, double im_scale,
    double P, double Q, int setType, double Z, double X, double C)
{   
    int pixPerThread = w * h / (gridDim.x * blockDim.x);
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    int offset = pixPerThread * tid;

    for (int i = offset; i < offset + pixPerThread; i++)
    {
        int x = i % w;
        int y = i / w;
        int m = 4* (x + y * w);
        double A = re_min + re_scale * x;
        double B = im_min + im_scale * y;
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

__device__ int mandelbrot(int iter, int max_iter, double A, double B, double P, double Q)
{
    while (iter < max_iter)
    {
        double tmp = (A*A) - (B*B) + P;
        B = (2 * A * B) + Q;
        A = tmp;
        if (A*A + B*B > 4)
            break;
        iter++;
    }
    return iter;
}

__device__ int burningShip(int iter, int max_iter, double A, double B, double P, double Q)
{
    while (iter < max_iter)
    {
        double tmp = (A * A) - (B * B) + P;
        B = fabsf(2 * A * B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
            break;
        iter++;
    }
    return iter;
}

__device__ int mandelCubed(int iter, int max_iter, double A, double B, double P, double Q)
{
    while (iter < max_iter)
    {
        double tmp = (A * A * A - 3 * A * B * B) + P;
        B = (-B * B * B + 3 * A * A * B) + Q;
        A = tmp;
        if (A * A + B * B > 4)
            break;
        iter++;
    }
    return iter;
}

__device__ int experimental(int iter, int max_iter, double A, double B, double s, double r, double f)
{   //Mandelbox
    double r_sq = r * r;
    double X = 0.0; double Y = 0.0;// float Z = 2.0;
    while (iter < max_iter-1)
    {
        double mag_sq = (X * X) + (Y * Y); // +(Z * Z);
        double mag = sqrtf(mag_sq);

        if (X > 1.0)
            X = 2.0 - X;
        else if (X < -1.0)
            X = -2.0 - X;

        if (Y > 1.0)
            Y = 2.0 - Y;
        else if (Y < -1.0)
            Y = -2.0 - Y;

        X *= f;
        Y *= f;

        if (mag < r)
        {
            X /= r_sq;
            Y /= r_sq;
        }
        else if (mag < 1.0)
        {
            X /= mag_sq;
            Y /= mag_sq;
        }

        X = X * s + A;
        Y = Y * s + B;

        if (mag > 4.0)
            break;
        iter++;
    }
    return iter;
}