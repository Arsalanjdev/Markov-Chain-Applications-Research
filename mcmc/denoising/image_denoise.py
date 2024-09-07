import sys
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def img_to_txt(path:str):

    image = Image.open(path).convert('L') #Graryscale (Black and white)
    image = np.asarray(image) #storing it into an array
    image = 2*(image > 128).astype(int)-1 #intensity and normilizing

    #printing pixels
    file = open(f"{path}.txt", "w")
    text = list(image)
    for i in text:
        for j in i:
            file.write(str(j) + " ")
        file.write('\n')
    file.close()

def txt_to_image(text_path):
    #image = np.loadtxt(text_path)
    with open(text_path, 'r') as file:
        lines = file.readlines()
        pixel_data = [[int(value) for value in line.split()] for line in lines]

    # Convert the pixel data to a NumPy array
    img_array = np.array(pixel_data, dtype=np.uint8)

    # Create an image from the pixel data
    img = Image.fromarray(img_array, 'L')  # 'L' mode for grayscale

    # Save the image
    img.save(f"{text_path}_img.PNG")

def noisify(path):
    img_to_txt(path)
    pixels = np.loadtxt(f"{path}.txt")

    # probability of random flips
    flip_probability = 0.15

    num_rows, num_columns = pixels.shape

    # Generate a random array for flips based on the defined probability
    random_flips = np.random.rand(num_rows, num_columns) < flip_probability

    # Apply random flips
    flipped_pixels = pixels * (-1) ** random_flips
    file = open(f"{path}_noisy.txt", "w")
    txt = list(flipped_pixels)

    #saving it
    for row in txt:
        for column in row:
            file.write(str(int(column)) + " ")
        file.write('\n')
    file.close()

    # converting to image
    txt_to_image(f"{path}_noisy.txt")

def denoisify(file,alpha,beta,iteration):

    # Prior belief parameters
    #0.8
    #0.16

    # Calculate gamma based on the prior belief
    gamma = 0.5 * np.log((1 - alpha) / alpha)

    # Number of steps for convergence
    img_to_txt(file)
    pixels = np.loadtxt(f"{file}.txt")
    num_rows, num_columns = pixels.shape
    # Initialize Z from X
    modified = pixels.copy()

    # Iterate for a given number of steps
    for t in range(iteration):
        # Randomly choose a coordinate (i, j)
        i, j = np.random.choice(num_rows), np.random.choice(num_columns)

        # Calculate acceptance probability

        delta = -2 * gamma * pixels[i, j] * modified[i, j] - 2 * beta * modified[i, j] * (
                np.sum(modified[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]) - modified[i, j]
        )

        # Flip the pixel if accepted
        if np.log(np.random.rand()) < delta:
            modified[i, j] = -modified[i, j]  # Update the image

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(pixels, cmap='gray', vmin=-1, vmax=1)
    ax[0].set_title("Noisy", fontsize=16)
    ax[1].imshow(modified, cmap='gray', vmin=-1, vmax=1)
    ax[1].set_title("Denoised", fontsize=16)
    plt.show()

def main():

    if input("Please enter a mode: D for de-noisifying an image"
             "N for Noisifying an image:\n").lower() == 'd':
        address = input("Please enter the address of your image:")
        alpha = float(input("Please enter a value for alpha parameter:"))
        beta = float(input("Please enter a value for beta parameter:"))
        iteration = int(input("Please enter a value for the number of iterations:"))
        denoisify(address, alpha, beta, iteration)
    else:
        address = input("Please enter the address of your image:")
        alpha = input("Please enter a value for alpha parameter:")
        noisify(address)
if __name__ == "__main__":
    main()
    #denoisify("a.jpg_noisy.txt_img.PNG")