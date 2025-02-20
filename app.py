import cv2
import mediapipe as mp
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

pg.init()
screen = pg.display.set_mode((640, 480), DOUBLEBUF | OPENGL)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
gluPerspective(45, (640 / 480), 0.1, 50.0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cube_size = 1.0
cube_x = 0.0
cube_y = 0.0
cube_z = -5.0
cube_rotation_x = 0.0
cube_rotation_y = 0.0
cube_rotation_z = 0.0

prev_cube_x = 0.0
prev_cube_y = 0.0
prev_cube_rotation_x = 0.0
prev_cube_rotation_y = 0.0
prev_cube_rotation_z = 0.0


def draw_cube(size):
    vertices = np.array([
        [-size, -size, -size],
        [size, -size, -size],
        [size, size, -size],
        [-size, size, -size],
        [-size, -size, size],
        [size, -size, size],
        [size, size, size],
        [-size, size, size]
    ])

    faces = [
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 3, 7], [0, 7, 4]
    ]

    colors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 0.0, 0.5]
    ]

    glBegin(GL_TRIANGLES)
    for i, face in enumerate(faces):
        glColor3f(*colors[i // 2])
        for vertex_index in face:
            vertex = vertices[vertex_index]
            glVertex3f(*vertex)
    glEnd()

def draw_axes(length):
    glLineWidth(2.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(length, 0.0, 0.0)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, length, 0.0)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, length)
    glEnd()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_hand_x = None
    left_hand_y = None
    right_hand_x = None
    right_hand_y = None


    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i == 0:  # Suponemos que la primera mano detectada es la izquierda
                left_hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                left_hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            elif i == 1: # Y la segunda es la derecha
                right_hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                right_hand_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    if left_hand_x is not None and left_hand_y is not None:
        cube_x = (left_hand_x - 0.5) * 5
        cube_y = -(left_hand_y - 0.5) * 5
        cube_x = prev_cube_x * 0.8 + cube_x * 0.2
        cube_y = prev_cube_y * 0.8 + cube_y * 0.2
        prev_cube_x = cube_x
        prev_cube_y = cube_y

    if right_hand_x is not None and right_hand_y is not None:
        cube_rotation_x = (right_hand_y - 0.5) * 360
        cube_rotation_y = (right_hand_x - 0.5) * 360
        cube_rotation_x = prev_cube_rotation_x * 0.8 + cube_rotation_x * 0.2
        cube_rotation_y = prev_cube_rotation_y * 0.8 + cube_rotation_y * 0.2

        prev_cube_rotation_x = cube_rotation_x
        prev_cube_rotation_y = cube_rotation_y




    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glPushMatrix()
    glTranslatef(-2, -2, -5)
    draw_axes(1)
    glPopMatrix()

    glTranslatef(cube_x, cube_y, cube_z)
    glRotatef(cube_rotation_x, 1, 0, 0)
    glRotatef(cube_rotation_y, 0, 1, 0)
    draw_cube(cube_size)


    cv2.imshow('Hand Tracking', frame)
    pg.display.flip()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
pg.quit()