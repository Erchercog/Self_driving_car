import numpy as np
import cv2
from directkeys import PressKey, W, A, S, D, ReleaseKey
import time


from grabscreen import grab_screen
from getkeys import key_check

from alexnet import alexnet

import os

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 12
MODEL_NAME = 'pygta5-car-800-600-{}-{}-{}-epochs'.format(LR, 'alexnetv2', EPOCHS)

t_time = 0.07
##p_time = 0.05

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(t_time)
##    ReleaseKey(W)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
##    for i in list(range(5))[::-1]:
##        print(i + 1)
##        time.sleep(1)

    j = g = 0
    paused = False
    print('START!!!')
    f_time = time.time()
    
    MAX_SPEED = 1.0
    s_time = time.time()
    while (True):
        
        if (not paused) and (s_time - f_time < MAX_SPEED):
            s_time = time.time()
            
            last_time = time.time()
            
            screen = grab_screen(region = (11, 30, 811, 630))

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            prediction = model.predict([screen.reshape(WIDTH, HEIGHT, 1)])[0]
            moves = list(np.around(prediction))
            FPS = str(60 / (time.time() - last_time)) + ' FPS'
            print(moves, prediction, FPS)

            if moves == [1, 0, 0]:
                left()
            elif moves == [0, 1, 0]:
                straight()
            elif moves == [0, 0, 1]:
                right()

        elif s_time - f_time >= MAX_SPEED:
            print('STOP!!!')

            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)

            time.sleep(MAX_SPEED)
            
            f_time = time.time()
            s_time = time.time()
            
            
            
        keys = key_check()

        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

        
        if 'Q' in keys:
            break


if __name__ == '__main__':
    main()
