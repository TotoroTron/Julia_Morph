# Julia_Morph
My first C++ project utilizing CUDA acceleration and SFML graphics. This is a basic application that visualizes the Julia Set for an arbitrary value of C in real time. On a GTX 1070 GPU, i5-4460 CPU, the application will generate 1920x1080 pixels at 60-80 frames per second. On worst case scenario, where the every coordinate on the screen is within the julia set, the FPS drops to 45. This is with a 1080 blocksPerGrid, 960 threadsPerBlock configuration. The iteration threshold for each coordinate is 240.
The user can shift the value of C on the complex plane to morph the Julia Set by using the WASD keys. The user can also recenter the image by clicking a new center using the left mouse button. You can also zoom in or out of the center by using the minus and equals keys. The controls have no effect on the FPS.