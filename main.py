import numpy as np
from PIL import ImageGrab
import cv2
from directkeys import PressKey, W, A, S, D, ReleaseKey
import time
import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from directkeys import PressKey, W, A, S, D
from statistics import mean


from grabscreen import grab_screen
from getkeys import key_check

import os


def keys_to_output(keys):
    #[A, W, D]
    output = [0, 0, 0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1

    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previos data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exists, starting fresh!')
    training_data = []



def main():
##    for i in list(range(5))[::-1]:
##        print(i + 1)
##        time.sleep(1)
##    last_time = time.time()
    printed = True
    j = g = 0
    print('START!!!')
    while (True):
        current_time = time.time()
        
        
        screen = grab_screen(region = (10, 27, 809, 626))

        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (160, 120))
        
        if printed:
            print('screen_resize: ',   screen.shape, 'type: ', type(screen[0][0]))
            printed = False

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen, output])

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

##        new_screen, original_img, m1, m2 = process_img(np.array(screen))

##        cv2.imshow('Canny', new_screen)
##        cv2.imshow('GTA V Neuro', cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

##        if m1 < 0 and m2 < 0:
##            rigth()
##        elif m1 > 0 and m2 > 0:
##            left()
##        else:
##            straight()
##        
##        
##        if cv2.waitKey(25) & 0xFF == ord('q'):
##            cv2.destroyAllWindows()
##            break
        if 'Q' in keys:
            print('FINISH!')
            break


if __name__ == '__main__':
    main()
