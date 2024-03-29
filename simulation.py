import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import sys

'''
* This program visualizes the N-body simulation from a dataset
'''

def get_particle_pos(line):
    particle_pos = np.array(line.replace("[", "").replace("]", "").replace(",", "").split()).astype(float)
    N = int(len(particle_pos) / 3) # each particle has (x, y, z)
    particle_pos = particle_pos.reshape(N, 3)
    return particle_pos

def init_animation():
    global scat
    particle_pos = get_particle_pos(first_line)
    scat = ax1.scatter(particle_pos[:, 0], particle_pos[:, 1], s=100*particle_mass, color = 'lime', edgecolor='purple')

def animate(i):
    # Get particle positions
    line = all_lines[i]
    particle_pos = get_particle_pos(line)
    offsets = np.asarray([particle_pos[:, 0], particle_pos[:, 1]]).T
    scat.set_offsets(offsets)

def main():
    # get file paths
    data_path = sys.argv[1]
    output_path = sys.argv[2]

    # Defile global variables
    global particle_mass, ax1, first_line, all_lines

    # Read data file
    with open(data_path, 'r') as f:
        content = f.readlines()

    # Recreate N x 1 array of particle masses
    masses = content[1].replace("[", "").replace("]", "").replace(",", "").split()
    
    particle_mass = np.array(masses).astype(float)
    
    # Visulization setup
    # Initialize plots
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor("blue")
    ax1.set(xlim=(-3, 3), ylim=(-3, 3))
    ax1.set_aspect('equal', 'box')
    plt.title(label= "N-Body Simulation", fontsize=20, color='black')

    first_line = content[3] # to plot the first frame
    all_lines = content[4:]
    
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=50, repeat_delay=100, cache_frame_data=False, init_func=init_animation)
    ani.save(output_path, writer='imagemagick', fps=30)

main()
