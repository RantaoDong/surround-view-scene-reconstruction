import cv2
import numpy as np
import pygame


def carla_image_to_bgr(image):
    pixels = np.frombuffer(image.raw_data, dtype=np.uint8)
    pixels = pixels.reshape((image.height, image.width, 4))
    return pixels[:, :, :3]


def draw_camera_image(surface, image, position):
    bgr_image = carla_image_to_bgr(image)
    height, width = bgr_image.shape[:2]
    preview = cv2.resize(bgr_image, (width // 4, height // 4))
    rgb_image = preview[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))
    surface.blit(image_surface, position)


def get_display_font():
    fonts = pygame.font.get_fonts()
    font_name = "ubuntumono" if "ubuntumono" in fonts else fonts[0]
    return pygame.font.Font(pygame.font.match_font(font_name), 14)


def exit_requested():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            return True
    return False

