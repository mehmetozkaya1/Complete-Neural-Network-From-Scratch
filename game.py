# Implementing necessary libraries
import pygame
import sys
import numpy as np
from PIL import Image, ImageDraw
from neuralNetwork import NeuralNetwork
from layer import Layer
from data_c import x_train, y_train

# Layers
layer1 = Layer(units=100, activation="relu", input_size=784)
layer2 = Layer(units=50, activation="relu", input_size=100)
layer3 = Layer(units=10, activation="sigmoid", input_size=50)
layers = [layer1, layer2, layer3]

# Neural network
epochs = 20
learning_rate = 0.03
cost_func = "MSE"

nn = NeuralNetwork(layers, x_train, y_train, cost_func)
cost_his = nn.train(epochs, learning_rate)

# Ekran boyutları
WIDTH, HEIGHT = 400, 400

# Renkler
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

def draw_button(screen, x, y, width, height, color, text):
    pygame.draw.rect(screen, color, (x, y, width, height))
    font = pygame.font.SysFont(None, 36)
    text_surface = font.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=(x + width / 2, y + height / 2))
    screen.blit(text_surface, text_rect)

def draw_text_input(screen, x, y, width, height, text):
    pygame.draw.rect(screen, WHITE, (x, y, width, height), 2)
    font = pygame.font.SysFont(None, 36)
    text_surface = font.render(text, True, BLACK)
    screen.blit(text_surface, (x + 5, y + 5))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Çizim Ekranı")
    clock = pygame.time.Clock()

    button1_width, button1_height = 100, 50
    button1_x, button1_y = 20, 330

    button2_width, button2_height = 100, 50
    button2_x, button2_y = 150, 330

    text_input_width, text_input_height = 100, 50
    text_input_x, text_input_y = WIDTH - text_input_width - 20, HEIGHT - text_input_height - 20

    drawing = False
    points = []
    text_input = ""

    while True:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Sol fare tuşuna basıldı
                    mouse_x, mouse_y = event.pos
                    if button1_x <= mouse_x <= button1_x + button1_width and button1_y <= mouse_y <= button1_y + button1_height:
                        print("Tahmin et!")
                        img_array = predict_number(points)  # Çizilen şekli tahmin et
                        # Predict
                        y_pred = nn.predict([img_array])
                        number = [np.argmax(pred) for pred in y_pred]
                        text_input = str(number)
                        print(number)
                    elif button2_x <= mouse_x <= button2_x + button2_width and button2_y <= mouse_y <= button2_y + button2_height:
                        print("Temizlendi!")
                        drawing = False
                        points.clear()  # Çizilen noktaları temizle
                    else:
                        drawing = True
                        points.append(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Sol fare tuşu bırakıldı
                    drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    points.append(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    text_input = text_input[:-1]  # Geri tuşuna basıldığında metni sil
                else:
                    text_input += event.unicode  # Klavyeden gelen karakteri metin kutusuna ekle

        draw_button(screen, button1_x, button1_y, button1_width, button1_height, RED, "Tahmin")
        draw_button(screen, button2_x, button2_y, button2_width, button2_height, RED, "Temizle")
        draw_text_input(screen, text_input_x, text_input_y, text_input_width, text_input_height, text_input)

        if len(points) > 1:
            pygame.draw.lines(screen, WHITE, False, points, 15)

        pygame.display.flip()
        clock.tick(60)

def predict_number(points):
    # Görüntü oluştur
    image = Image.new('RGB', (WIDTH, HEIGHT), BLACK)
    draw = ImageDraw.Draw(image)
    draw.line(points, fill=WHITE, width=3)

    # Görüntüyü boyutlandır ve dönüştür
    image = image.resize((28, 28)).convert('L')  # 28x28 boyutuna dönüştür ve siyah-beyaz yap
    image_array = np.array(image) / 255.0  # Normalizasyon
    image_array = image_array.reshape(784, 1)

    return image_array

main()