import sys
sys.path.append("../src")
from network import Network
from layers.dense import Dense

net = Network()

net.add_layers(
    Dense((784, 80), activation="relu"),
    Dense((80, 80), activation="relu"),
    Dense((80, 10), activation="softmax"),
)

net.load_model("network.json")

import pygame
import numpy as np
from PIL import Image

pygame.init()
screen = pygame.display.set_mode((648, 448))
drawing_board = pygame.Surface((448, 448))

running = True

screen.fill((0, 0, 0))
drawing_board.fill((0, 0, 0))
pygame.draw.rect(
    screen,
    (100, 100, 100),
    (448, 0, 200, 448)
)

def center_image(image):
    rows, cols = np.indices(image.shape)
    total_mass = np.sum(image)
    center_of_mass_row = np.sum(rows * image) / total_mass
    center_of_mass_col = np.sum(cols * image) / total_mass
    
    center_row, center_col = np.array(image.shape) // 2
    shift_row = center_row - center_of_mass_row
    shift_col = center_col - center_of_mass_col
    
    centered_image = np.roll(image, int(shift_row), axis=0)
    centered_image = np.roll(centered_image, int(shift_col), axis=1)
    
    return centered_image

title_font = pygame.font.SysFont("arial", 30)
body_font = pygame.font.SysFont("arial", 20)

img = []
def evaluate_image():
    global img
    img = pygame.surfarray.array2d(drawing_board)
    img = (img / 16777215)
    img = img.reshape(28, 16, 28, 16)
    img = np.mean(img, axis=(1, 3))
    img = img.transpose()    
    img = center_image(img)

    img = img.reshape(784, 1)
    results = net.feedforward(img[np.newaxis, ...])[0]

    pygame.draw.rect(
        screen,
        (100, 100, 100),
        (448, 0, 200, 448)
    )   
    surface = title_font.render(f"Prediction: {np.argmax(results)}", True, (0, 0, 0))
    screen.blit(surface, (448 + 4, 0 + 4))

    for x in range(10):
        surface = body_font.render(f"{x}: {round(results[x][0] * 100, 2)}%", True, (0, 0, 0))
        screen.blit(surface, (448 + 4, 0 + 40 + x * 20 + 2))

prev_pos = (-1, -1)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DELETE:
                drawing_board.fill((0, 0, 0))
            if event.key == pygame.K_e:
                Image.fromarray(np.uint8(img.reshape(28, 28) * 255), mode="L").show()
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            if (prev_pos == (-1, -1)): 
                prev_pos = (x, y)
                continue
            pygame.draw.line(drawing_board, (255, 255, 255), (prev_pos[0], prev_pos[1]), (x, y), 20)
            pygame.draw.circle(drawing_board, (255, 255, 255), (x, y), 9)
            prev_pos = (x, y)
            evaluate_image()

        else: 
            prev_pos = (-1, -1)

    screen.blit(drawing_board, (0, 0))
    pygame.display.flip()
    
pygame.quit()