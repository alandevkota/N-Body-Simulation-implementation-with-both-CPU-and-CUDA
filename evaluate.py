import sys
import os
import matplotlib.pyplot as plt

'''
* This code is use to generate a graph to compare the runtine of the serial and parallel 
implemetations of the N-body simulation.
* I have chosen N to increase in power of N and number of timsteps constant as (150).
'''
def plotthegraph(py, cu, n_vals):
    plt.xscale("log")
    plt.title(label="Runtime Comparision", fontsize=20, color='black')
    plt.plot(n_vals, py, label="Serial Implementation")
    plt.plot(n_vals, cu, label="Parallel Implementation")
    plt.xlabel('Number of Particles (log scale)')
    plt.ylabel('Runtime (sec)')
    plt.legend()
    plt.savefig("Evaluate_" + str(len(n_vals)) + "_iter.png")

def simulationrun(N, itr):
    py_runtimes = []
    cu_runtimes = []
    n_vals = []
    for i in range(itr):
        n_vals.append(N)
        print("Running evaluation with N = " + str(N))

        # Get serial nbody runtime
        os.system("python nbody.py " + str(N) + " 150")
        f_py = open("output_py.txt", 'r')
        content_py = f_py.readlines()
        py_runtimes.append(float(content_py[2]))
        f_py.close()

        # Get parallel nbody runtime
        # os.chdir("../build")
        os.system("./nbody " + str(N) + " 150")
        # os.chdir("../python_code")
        f_cu = open("output_cu.txt", 'r')
        content_cu = f_cu.readlines()
        cu_runtimes.append(float(content_cu[2]))
        f_cu.close()
        
        # Increment number of N (powers of 2)
        N *= 2
        
    return py_runtimes, cu_runtimes, n_vals



def main():
    ''' Run Evaluation of Serial vs Parallel Implementaitons '''

    itr = int(sys.argv[1])
    py, cu, n_vals= simulationrun(2, itr)
    plotthegraph(py, cu, n_vals)

main()
