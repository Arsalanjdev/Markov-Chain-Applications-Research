import numpy as np
from scipy import linalg as sp
import numpy.random as rand
import matplotlib.pyplot as plot


def load_chain(fileName, d):
    matrix = np.load(fileName)

    matrix *= 1 - d
    matrix += d / matrix.shape[0]

    vector = np.arange(0, matrix.shape[0])
    return vector, matrix


M = load_chain('main.npy', 0.11)

rand.seed(42)

#print('Number of states:', len(M[0]))
#print('Random state:', M[0][rand.randint(len(M[0]))])
#M = load_chain('example.npy', 0.11)
print('Number of states:', len(M[0]))
print('Transition probabilities:')
print(M[1])


def prob_trajectory(chain, traj):
    result = 1
    P = chain[1]

    for i in range(len(traj) - 1):
        result *= P[int(traj[i])][int(traj[i + 1])]

    return result


print('Prob. of trajectory (1, 3, 8):', prob_trajectory(M, ('1', '3', '8')))


np.set_printoptions(precision=3)


def stationary_dist(mc):
    P = mc[1]

    eigenvalues, eigenvectors = sp.eig(P, left=True, right=False)

    index = 0

    for i, value in enumerate(eigenvalues):
        if (abs(1 - value) < 1e-10):
            index = i

    result = eigenvectors[:, index].real  #real part only
    totalSum = sum(result)
    result /= totalSum  # normalize

    return result


u_star = stationary_dist(M)

print('Stationary distribution:')
print(u_star)

u_prime = u_star.dot(M[1])

print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))


def compute_dist (Markov,initial,number_of_steps):
    P = Markov[1]
    for i in range(number_of_steps):
        initial = np.dot(initial,P)
    return initial

#M = load_chain('example.npy', 0.11)
u_star = stationary_dist(M)

# Number of states
nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

# Distrbution after 100 steps
v = compute_dist(M, u, 10)
print('\nIs u * P^10 = u*?', np.all(np.isclose(v, u_star)))

# Distrbution after 1000 steps
v = compute_dist(M, u, 100)
print('\nIs u * P^100 = u*?', np.all(np.isclose(v, u_star)))


def simulate(markov, initial, number_of_steps):
    numStates = len(markov[0]) #number of states in markovchain
    P = markov[1] #transition matrix
    traj = [None] * number_of_steps #storing the trajectory
    state = np.random.choice(numStates, 1, p=initial[0])[0]

    for i in range(number_of_steps):
        traj[i] = str(state)
        nextState = np.random.choice(numStates, 1, p=P[state])[0]
        state = nextState

    return tuple(traj)


# Number of states
nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

# Simulate short trajectory
traj = simulate(M, u, 10)
print(traj)

# Simulate a long trajectory
traj = simulate(M, u, 10000)

def histogram(traj, uStar):
    intTraj = tuple(map(int, traj))

    plot.scatter(np.arange(nS), uStar, color='red', zorder=2)
    plot.hist(intTraj, align='left', rwidth=0.5, bins=np.arange(nS + 1), density=True, zorder=1)
    plot.show()

print(np.arange(nS))
histogram(traj, u_star)
