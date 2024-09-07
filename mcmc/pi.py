import random
import matplotlib.pyplot as plt

def monte_carlo_pi_approximation(num_points):
    points_inside_circle = 0
    inside_circle_x = []
    inside_circle_y = []
    outside_circle_x = []
    outside_circle_y = []

    for _ in range(num_points):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        distance = x**2 + y**2

        if distance <= 1:
            points_inside_circle += 1
            inside_circle_x.append(x)
            inside_circle_y.append(y)
        else:
            outside_circle_x.append(x)
            outside_circle_y.append(y)

    pi_approximation = 4 * (points_inside_circle / num_points)
    return pi_approximation, inside_circle_x, inside_circle_y, outside_circle_x, outside_circle_y

num_points = 20000
result, inside_x, inside_y, outside_x, outside_y = monte_carlo_pi_approximation(num_points)

plt.figure(figsize=(8, 8))
plt.scatter(inside_x, inside_y, color='blue', label='Inside Circle')
plt.scatter(outside_x, outside_y, color='red', label='Outside Circle')
plt.title(f'Monte Carlo estimation of π: {result:.6f} using {num_points} points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.axis('equal')
plt.show()
