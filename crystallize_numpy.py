#!/usr/bin/env python3
import pygame
import math
import numpy as np

WIDTH = 960
HEIGHT = 960
FPS = 1000

BG = "#333333"

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Crystals")
clock = pygame.time.Clock()

font = pygame.font.Font(None, 30)
TEXT_COLOR = (255, 255, 255)

camera_posx, camera_posy = 0, 0
camera_zoom = 1
dragging = False

DT = 0.01

WORLD_CENTER = (WIDTH / 2, HEIGHT / 2)
K = 0.003

PART1_COL    = "#5555ee"
PART1_Q      = 1
PART1_SIZE   = 10
PART1_WEIGHT = 1

PART2_COL   = "#ee5555"
PART2_Q     = -1
PART2_SIZE  = 10
PART2_WEIGHT = 1

_GCD = math.gcd(PART1_WEIGHT, PART2_WEIGHT)
PART1_WEIGHT //= _GCD
PART2_WEIGHT //= _GCD
PARTICLE_TYPES = [(PART1_Q, PART1_SIZE, PART1_COL)] * PART1_WEIGHT +\
                 [(PART2_Q, PART2_SIZE, PART2_COL)] * PART2_WEIGHT

DRAG = 0.5
TEMP = 70

np.random.seed(42)

MAX_PARTICLES = 260
ADD_PARTICLE_MS = 1000
ADD_PARTICLE_ANGLE_ADVANCE = 0.4

pos, vel, acc, charges, sizes, colors = np.zeros((MAX_PARTICLES, 2), dtype=np.float64), np.zeros((MAX_PARTICLES, 2), dtype=np.float64),\
                                        np.zeros((MAX_PARTICLES, 2), dtype=np.float64), np.zeros(MAX_PARTICLES, dtype=np.float64),\
                                        np.zeros(MAX_PARTICLES, dtype=np.float64), []
N = 0

ADD_PARTICLE_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(ADD_PARTICLE_EVENT, ADD_PARTICLE_MS)
ELLIPSE_A = WIDTH / 2
ELLIPSE_B = HEIGHT / 2
ANGLE = 0

running = True
frame_count = 0

fps_surf = font.render(f"FPS: 0 (0 real)", True, TEXT_COLOR)
temp_surf = font.render(f"TEMP: 0", True, TEXT_COLOR)
cam_surf = font.render(f"CAM: zoom: 0; pos: (0, 0)", True, TEXT_COLOR)
part_surf = font.render(f"PARTS: 0/0", True, TEXT_COLOR)

while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                dragging = True
            elif event.button in (4, 5):
                mouse_x, mouse_y = pygame.mouse.get_pos()
                world_x = mouse_x / camera_zoom + camera_posx
                world_y = mouse_y / camera_zoom + camera_posy

                zoom_speed = 1.1
                if event.button == 4:
                    camera_zoom *= zoom_speed
                elif event.button == 5:
                    camera_zoom /= zoom_speed

                camera_zoom = max(0.1, min(camera_zoom, 10.0))
                camera_posx = world_x - mouse_x / camera_zoom
                camera_posy = world_y - mouse_y / camera_zoom

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if dragging:
                mouse_dx, mouse_dy = event.rel
                camera_posx -= mouse_dx / camera_zoom
                camera_posy -= mouse_dy / camera_zoom

        elif event.type == ADD_PARTICLE_EVENT:
            if N < MAX_PARTICLES:
                ANGLE += ADD_PARTICLE_ANGLE_ADVANCE
                ANGLE %= 2 * math.pi
                x = WORLD_CENTER[0] + ELLIPSE_A * math.cos(ANGLE)
                y = WORLD_CENTER[1] + ELLIPSE_B * math.sin(ANGLE)
                pos[N] = [x, y]
                particle_type = PARTICLE_TYPES[N % len(PARTICLE_TYPES)]
                colors += [particle_type[2]]
                sizes[N] = particle_type[1]
                charges[N] = particle_type[0]
                N += 1

    if frame_count % 2 == 0:
        screen.fill(BG)

    for _ in range(8):
        diff = pos[:N, np.newaxis, :] - pos[np.newaxis, :N, :]

        r_sq = np.sum(diff**2, axis=-1)

        np.fill_diagonal(r_sq, np.inf)

        sizes_sum = sizes[:N, np.newaxis] + sizes[np.newaxis, :N]
        sigma = sizes_sum * 0.890898718147

        charges_prod = charges[:N, np.newaxis] * charges[np.newaxis, :N]
        epsilon = 400

        min_r_sq = (0.4 * sigma)**2
        r_sq_clamped = np.maximum(r_sq, min_r_sq)

        r = np.sqrt(r_sq_clamped)
        sr = sigma / r
        sr6 = sr**6
        sr12 = sr6**2

        F_r = 24 * epsilon * (2 * sr12 - sr6) / r_sq_clamped

        is_similar = charges_prod > 0
        F_r = np.where(is_similar, 48 * epsilon * sr12 / r_sq_clamped, F_r)

        force_scalar = F_r

        forces = diff * force_scalar[:, :, np.newaxis]
        acc = np.sum(forces, axis=1)

        d_center = pos[:N] - WORLD_CENTER
        center_force = -K * d_center
        acc += center_force

        vel[:N] += acc * DT
        vel[:N] *= (1.0 - DRAG * DT)

        if TEMP > 0:
            noise_scale = math.sqrt(2 * TEMP * DRAG * DT)
            vel[:N] += np.random.normal(0, 1, (N, 2)) * noise_scale

        pos[:N] += vel[:N] * DT

    # # Annealing
    # if TEMP > 0.01:
    #     TEMP *= 0.99999
    # else:
    #     TEMP = 0

    drawx = ((pos[:N, 0] - camera_posx) * camera_zoom).astype(int)
    drawy = ((pos[:N, 1] - camera_posy) * camera_zoom).astype(int)
    draw_sizes = np.round(sizes * camera_zoom).astype(int)

    for i in range(N):
        if -draw_sizes[i] <= drawx[i] <= WIDTH + draw_sizes[i] and -draw_sizes[i] <= drawy[i] <= HEIGHT + draw_sizes[i]:
            pygame.draw.circle(screen, colors[i], (drawx[i], drawy[i]), draw_sizes[i], width=0)

    if frame_count % 20 == 0:
        fps = int(clock.get_fps())
        fps_surf = font.render(f"FPS: {fps} ({int(fps / 2)} real)", True, TEXT_COLOR)
        temp_surf = font.render(f"TEMP: {TEMP:.2f}", True, TEXT_COLOR)
        cam_surf = font.render(f"CAM: zoom: {camera_zoom:.2f}; pos: ({camera_posx:.2f}, {camera_posy:.2f})", True, TEXT_COLOR)
        part_surf = font.render(f"PARTS: {N}/{MAX_PARTICLES}", True, TEXT_COLOR)

    if frame_count % 2 == 0:
        screen.blit(fps_surf, (10, 10))
        screen.blit(temp_surf, (10, 40))
        screen.blit(cam_surf, (10, 70))
        screen.blit(part_surf, (10, 100))

        pygame.draw.circle(screen, "#ffff00", ((WORLD_CENTER[0] - camera_posx) * camera_zoom, (WORLD_CENTER[1] - camera_posy) * camera_zoom), 2, width=0)

        pygame.display.flip()
    frame_count += 1

pygame.quit()
