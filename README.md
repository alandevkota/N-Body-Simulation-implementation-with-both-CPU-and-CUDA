# Steps to run the nbody simulation: 

First, the position of
N particles is generated over a given number of timesteps and
stored in an output text file for both CPU and GPU implementations.
The nbody.py is the python CPU-based program that
creates an output py.txt text file. N and timesteps are command
line arguments illustrated in the example below:


*   python nbody.py 1000 150

The nbody.cu is the GPU-based program compiled to create
an executable file called nbody. Executing nbody creates an
output cu.txt text file. N and timesteps are command line
arguments illustrated in the example below:


*   ./nbody 1000 150

Next, the movement of the particle positions generated from
the python and Cuda programs is plotted using a simulator
program written in python. This program’s input file is
supplied as a command line option for both CPU and GPU
implementations illustrated in the examples below:


*   python simulation.py output py.txt anim py.gif
*   python simulation.py output cu.txt anim cu.gif


Note: To create gif files for the simulation we need to install
imagemagick library using ! apt install imagemagick.
Finally, the runtime of both CPU and GPU implementations
are plotted by increasing the number of bodies in powers of 2
via an evaluation program written in python. The number of
iterations (times the number of bodies increases) is supplied
as a command line input illustrated in the example below:


*   python evaluate.py 7

The example in the code is running the evaluation with iterations=7.


# Link to directly run in google colab:

https://colab.research.google.com/drive/1V8g7oCyp1UxGWFdku9eMn44U1LkSuZdf?usp=sharing

# Note: Read GPU_project_Report.pdf for more information


