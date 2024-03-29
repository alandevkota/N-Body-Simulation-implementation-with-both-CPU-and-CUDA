import numpy as np
import math
import time
import matplotlib.pyplot as plt 
import sys

''' 
* This program generates the movement of particles using the N-body problem. 
* Produces an output file with particles' positions over a given number of timesteps.
'''

# Return matrix of accelertaions (x, y, z) for each particle
# p: Maxtix of particle postions (x, y, z)
# m: Array of particle masses
# G: Newton's Gravitational Constant
# N: Number of particles (bodies)
def getAcc(p, m, G, N):
    # new_acc is N x 3 matrix of updated accelerations (x, y, z for each particle)
    new_acc = np.zeros((len(p), 3))
    
    # get accleration for each particle
    for i in range(N):
        x = 0
        y = 0
        z = 0
        # Calculate orce exerted on each particel THEN sum
        for j in range(N):
            if j != i:
                # Get difference in position of neighboring particle
                dx = p[j][0] - p[i][0]
                dy = p[j][1] - p[i][1]
                dz = p[j][2] - p[i][2]

                # Calculate inverse with softening length (0.1) -- Part to account for particles close to eachother
                inv = (dx**2 + dy**2 + dz**2 + 0.1**2)**(-1.5)

                # Update acceleration (x, y, z)
                x += m[j] * dx / inv
                y += m[j] * dy / inv
                z += m[j] * dy / inv
        
        # Ajust with Newton's gravitational constant
        x *= G
        y *= G
        z *= G

        # Update with new acceleration
        new_acc[i][0] = x
        new_acc[i][1] = y
        new_acc[i][2] = z

    return new_acc

# Return array with postions of particles for each timestep
# data: array of maxtix of postions of particles for each timestep
# p: particle postions (x, y, z)
def format(data, p):
    all_pos = []
    for i in range(len(p)):
        for j in range(3):
            all_pos.append(p[i][j])
    data.append(all_pos)
    
    return data

# Write data to output file
# data: Maxtix of postions of particles for each timestep of the simulation
# output_file: name of output file
def generateOutput(data, output_file, runtime):
    f = open(output_file, 'a')
    f.write(str(runtime) + "\n")
    for i in range(len(data)):
        f.write(str(data[i]) + "\n")
    f.close()


def main():
    ''' Generate N-body Simulation Data '''

    # Number of particles
    N = int(sys.argv[1])
    # Newton's Gravitational Constant
    G = 6.67 * 10**-11
    # Random number generator seed
    np.random.seed(811)
    # Path for output file
    output = "output_py.txt"

    # Start timer -> for performance comparision
    t_start = time.time()

    # Create N x 3 matrix of random starting postion of particles (size N) -> each partile has x,y,z corrdinate
    particle_pos = np.random.uniform(-2.5, 2.5, N *3)
    particle_pos = np.reshape(particle_pos, (-1,3))

    # Data that will be outputed
    data = format([], particle_pos)

    # Create N x 3 matrix of random starting velocities for each particle
    particle_vel = np.random.randn(N, 3)

    # Create N x 1 array of masses of particles
    particle_mass = np.random.rand(N,1)

    # Get starting accelerations of particles (N x 3 matrix)
    particle_acc = getAcc(particle_pos, particle_mass, G, N)

    # Set number of timesteps (number of interations for simulation)
    td = 0.01 # Timestep duration
    timesteps = int(sys.argv[2]) # Number of timesteps

    f = open(output, 'w+')
    f.write("Positions of " + str(N) + " particles over " + str(timesteps) + " timesteps: \n")
    f.write(str(particle_mass.tolist()) + "\n")
    f.close()

    # Loop for number of timesetps
    for i in range(timesteps): # change 5 to timesteps
        # Leapfrog integration
        # 1) first half kick
        particle_vel += particle_acc * (td / 2.0)

        # 2) Drift --> Update positions of all particles
        particle_pos += particle_vel * td

        # 3) Get new accleration for each particle
        particle_acc = getAcc(particle_pos, particle_mass, G, N)

        # 4) Second half of kick --> update velocities
        particle_vel += particle_acc * (td / 2.0)

        # 6) Append to data that will be outputed
        data = format(data, particle_pos)

    # Get end time of simuation
    t_end = time.time()

    # Write to output file
    generateOutput(data, output, t_end - t_start)

main()
