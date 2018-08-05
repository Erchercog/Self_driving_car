import numpy as np
from PIL import ImageGrab
import cv2
from getkeys import key_check

def screen_record():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('testing_neuro_net.avi', fourcc, 20.0, (1920, 1080))

    print('RECORD START!')
    while(True):
        frame =  cv2.cvtColor(np.array(ImageGrab.grab()), cv2.COLOR_BGR2RGB)

        out.write(frame)
        
##        cv2.imshow('window',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & ('Q' in key_check()):
            print('RECORD ENDS!')
            cv2.destroyAllWindows()
            break

screen_record()
