import numpy        as     np
from   PIL          import ImageGrab
import cv2
from   directkeys   import PressKey, W, A, S, D, ReleaseKey
import time
import pyautogui
from   numpy        import ones,vstack
from   numpy.linalg import lstsq
from   directkeys   import PressKey, W, A, S, D
from   statistics   import mean


from   grabscreen   import grab_screen
from   getkeys      import key_check

import os

###########################################################################

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, [255, 255, 255])
    masked = cv2.bitwise_and(img, mask)
    return masked

###########################################################################

##def draw_lanes(img, lines, color = [0, 255, 255], thickness = 3):
##
##    # if this fails, go with some default line
##    try:
##
##        # finds the maximum y value for a lane marker 
##        # (since we cannot assume the horizon will always be at the same point.)
##
##        ys = []  
##        for i in lines:
##            for ii in i:
##                ys += [ii[1],ii[3]]
##        min_y = min(ys)
##        max_y = 600
##        new_lines = []
##        line_dict = {}
##
##        for idx,i in enumerate(lines):
##            for xyxy in i:
##                # These four lines:
##                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
##                # Used to calculate the definition of a line, given two sets of coords.
##                x_coords = (xyxy[0], xyxy[2])
##                y_coords = (xyxy[1], xyxy[3])
##                A = vstack([x_coords,ones(len(x_coords))]).T
##                m, b = lstsq(A, y_coords)[0]
##
##                # Calculating our new, and improved, xs
##                x1 = (min_y-b) / m
##                x2 = (max_y-b) / m
##
##                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
##                new_lines.append([int(x1), min_y, int(x2), max_y])
##
##        final_lanes = {}
##
##        for idx in line_dict:
##            final_lanes_copy = final_lanes.copy()
##            m =     line_dict[idx][0]
##            b =     line_dict[idx][1]
##            line =  line_dict[idx][2]
##            
##            if len(final_lanes) == 0:
##                final_lanes[m] = [ [m, b, line] ]
##                
##            else:
##                found_copy = False
##
##                for other_ms in final_lanes_copy:
##
##                    if not found_copy:
##                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
##                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
##                                final_lanes[other_ms].append([m, b, line])
##                                found_copy = True
##                                break
##                        else:
##                            final_lanes[m] = [ [m, b, line] ]
##
##        line_counter = {}
##
##        for lanes in final_lanes:
##            line_counter[lanes] = len(final_lanes[lanes])
##
##        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]
##
##        lane1_id = top_lanes[0][0]
##        lane2_id = top_lanes[1][0]
##
##        def average_lane(lane_data):
##            x1s = []
##            y1s = []
##            x2s = []
##            y2s = []
##            for data in lane_data:
##                x1s.append(data[2][0])
##                y1s.append(data[2][1])
##                x2s.append(data[2][2])
##                y2s.append(data[2][3])
##            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 
##
##        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
##        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])
##
##        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
##    except Exception as e:
####        print(str(e))
##        pass

###########################################################################

def gimp_to_opencv_hsv(*hsv):
    return (hsv[0] / 2, hsv[1] / 100 * 255, hsv[2] / 100 * 255)

###########################################################################

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

###########################################################################

WHITE_LINES = { 'low_th':   gimp_to_opencv_hsv(0,   0,   60),
                'high_th':  gimp_to_opencv_hsv(359, 30,  100) }

YELLOW_LINES = { 'low_th':  gimp_to_opencv_hsv(155, 30,  40),
                 'high_th': gimp_to_opencv_hsv(205, 100, 100),
                 'kernel':  np.ones((3,3), np.uint64)}



##WHITE_LINES_MIN = np.array([0, 0, 60], np.uint8)
##WHITE_LINES_MAX = np.array([359, 30, 100], np.uint8)
##
##YELLOW_LINES_MIN = np.array([155, 30, 40], np.uint8)
##YELLOW_LINES_MAX = np.array([205, 100, 100], np.uint8)

ST_ROAD_MIN = np.array([0, 0, 51],     np.uint8)
ST_ROAD_MAX = np.array([255, 51, 240], np.uint8)



###########################################################################

def get_lane_lines_mask(hsv_image, colors):
    masks = []
    for color in colors:
        if 'low_th' in color and 'high_th' in color:
            mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
            if 'kernel' in color:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color['kernel'])
            masks.append(mask)
        else: raise Exception('High or low threshold values missing')
    if masks:
        return cv2.add(*masks)
    
###########################################################################
    
def draw_binary_mask(binary_mask, img):
    if len(binary_mask.shape) != 2: 
        raise Exception('binary_mask: not a 1-channel mask. Shape: {}'.format(str(binary_mask.shape)))
    masked_image = np.zeros_like(img)
    for i in range(3): 
        masked_image[:,:,i] = binary_mask.copy()
    return masked_image

###########################################################################

##def new_road_colors(DY_ROAD_CLR_VERTICES_L, DY_ROAD_CLR_VERTICES_R, hsv):
##    DY_ROAD_CLR_ROI_L = roi(hsv, [DY_ROAD_CLR_VERTICES_L])
##    DY_ROAD_CLR_ROI_R = roi(hsv, [DY_ROAD_CLR_VERTICES_R])
##
##    DY_ROAD_CLR_ROI = cv2.bitwise_or(DY_ROAD_CLR_ROI_L, DY_ROAD_CLR_ROI_R)
##    
##    
##    DY_ROAD_CLR_ROI_reshape = DY_ROAD_CLR_ROI.reshape(3, 600, 800)
##    print(hsv_reshape.shape)
##    h = hsv_reshape[0]
##    s = hsv_reshape[1]
##    v = hsv_reshape[2]
##    
##    img_sort_h = np.sort(h, axis = None)
##    img_sort_s = np.sort(s, axis = None)
##    img_sort_v = np.sort(v, axis = None)

    
    

###########################################################################
def color_road1(hsv,             DLCX, DLCY,   ULCX, ULCY,   URCX, URCY,   DRCX, DRCY,   G1X, G1Y,   G2X, G2Y, MIN_CLR = ST_ROAD_MIN, MAX_CLR = ST_ROAD_MAX):
    vertices_frame = np.array([[DLCX, DLCY], [ULCX, ULCY], [URCX, URCY], [DRCX, DRCY], [G1X, G1Y], [G2X, G2Y]], np.int32)
    mask_frame = roi(hsv, [vertices_frame])
    
    color_road_masc = cv2.inRange(mask_frame, MIN_CLR, MAX_CLR)
    return draw_binary_mask(color_road_masc,  mask_frame)

def color_road0(hsv,             DLCX, DLCY,   ULCX, ULCY,   URCX, URCY,   DRCX, DRCY, MIN_CLR = ST_ROAD_MIN, MAX_CLR = ST_ROAD_MAX):
    vertices_frame = np.array([[DLCX, DLCY], [ULCX, ULCY], [URCX, URCY], [DRCX, DRCY]], np.int32)
    mask_frame = roi(hsv, [vertices_frame])
    
    color_road_masc = cv2.inRange(mask_frame, MIN_CLR, MAX_CLR)
    return draw_binary_mask(color_road_masc,  mask_frame)

###########################################################################

##def process_img(frame, vertices_frame, hsv):
##    original_image = frame
##    
##    # convert to gray
####    processed_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##    
##    # edge detection
####    processed_img =  cv2.Canny(processed_img, threshold1 = 280, threshold2 = 360)
##        
##    color_lanes_mask = get_lane_lines_mask(hsv, [WHITE_LINES, YELLOW_LINES])
##    color_lanes_masked = draw_binary_mask(color_lanes_mask,   hsv)
##
##    processed_img = cv2.Canny(color_lanes_masked,   280,    360)
##    
##    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
##
##
##    processed_img = roi(processed_img, [vertices_frame])
##
##    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
##    #                                     rho   theta   thresh  min length, max gap:        
##    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 300,      1500,       15)
##    
##    BLACK = np.zeros((600, 800))
##    
##    try:
##        l1, l2 = draw_lanes(original_image, lines)
##        cv2.line(BLACK, (l1[0], l1[1]), (l1[2], l1[3]), [0, 255, 0], 30)
##        cv2.line(BLACK, (l2[0], l2[1]), (l2[2], l2[3]), [0, 255, 0], 30)
##    except Exception as e:
####        print(str(e))
##        pass
##    try:
##        for coords in lines:
##            coords = coords[0]
##            try:
##                cv2.line(BLACK, (coords[0], coords[1]), (coords[2], coords[3]), [255, 0, 0], 3)
##                
##                
##            except Exception as e:
####                print(str(e))
##                pass
##    except Exception as e:
##        pass
##    
##    return BLACK

###########################################################################

def LRF_drive(L, R, F, Flag, LR_time, max_time):
    drive = np.zeros(3, np.uint8)

    Lcount0 = 0
    Lcount1 = 0

    Rcount0 = 0
    Rcount1 = 0

    Fcount0 = 0
    Fcount1 = 0

    Lcount1 = np.count_nonzero(L == 255)
    Rcount1 = np.count_nonzero(R == 255)
    Fcount1 = np.count_nonzero(F == 255)

    LcountP = Lcount1 / 13689 *100
    RcountP = Rcount1 / 15405 *100
    FcountP = Fcount1 / 4061 *100
    
    print('L: ', LcountP, '%')
    print('R: ', RcountP, '%')
    print('F: ', FcountP, '%')

    if FcountP >= 25:
        if LcountP <= 50:
            LcountP = 0
            RcountP = 100
            FcountP = 0
        elif RcountP <= 50:
            LcountP = 100
            RcountP = 0
            FcountP = 0
        else:
            LcountP = 0
            RcountP = 0
            FcountP = 100
    else:
        if (LcountP >= 50) and (RcountP >= 50):
            FcountP = 0
        elif LcountP >= 50:
            LcountP = 100
            RcountP = 0
            FcountP = 0
        elif RcountP >= 50:
            LcountP = 0
            RcountP = 100
            FcountP = 0
        else:
            LcountP = 0
            RcountP = 0
            FcountP = 0
    
    print('L: ', LcountP, '%')
    print('R: ', RcountP, '%')
    print('F: ', FcountP, '%')
    ####
    if LcountP >= 50:
##        if LR_time >= max_time:
        drive[0] = LcountP
        left()
##            LR_time = time.time()
    if RcountP >= 50:
##        if LR_time >= max_time:
        drive[2] = RcountP
        right()
##            LR_time = time.time()
    if FcountP >= 50:
        drive[1] = FcountP
        straight()
    if ((LcountP < 50) and (RcountP < 50) and (FcountP < 50)) and Flag == False:
        slow_ya_roll()
        Flag = True

    print('drive: ', drive)
    return Flag, LR_time

t_time = 0.08

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    time.sleep(0.2)
    ReleaseKey(W)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(W)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(W)
    ReleaseKey(D)

def slow_ya_roll():
##    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

###########################################################################

def main():
    j = g = 0
    
    paused =  True
    printed = True
    print(                 'FILE START!')
    i = 1
    j = True
    g = 0
    Flag = False
    s_time = time.time()
    
    LR_time = 4.0
    
    while(True):
        keys = key_check()
        
            
##            if j:
##                frame_file_name =   'training_img-{}-data.npy'.format(i)
##                lane_file_name =    'training_lane-{}-data.npy'.format(i)
##                LRF_file_name =     'training_LRF-{}-data.npy'.format(i)
##                j = False
##
##            if not (os.path.isfile(frame_file_name)):
##                lane_training_data =    []
##                LRF_training_data =     []
##                frame_training_data =   []
            
        frame =  cv2.cvtColor(grab_screen(region = (10, 27, 809, 626)), cv2.COLOR_BGR2RGB)
            
        vertices_frame = np.array([[0, 600], [0, 349], [221, 293], [543, 288], [800, 331], [800, 600]], np.int32)
##        mask_frame = roi(road_frame, [vertices_frame])
        mask_frame = roi(frame, [vertices_frame])

        hsv = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2HSV)

        hsv = cv2.GaussianBlur(hsv, (5, 5), 75, 75)

##        DY_ROAD_CLR_VERTICES_L = np.array([[61, 578], [94, 442], [198, 425], [177, 582]], np.int32)
##        DY_ROAD_CLR_ROI_L = roi(hsv, [DY_ROAD_CLR_VERTICES_L])

##        DY_ROAD_CLR_VERTICES_R = np.array([[697, 579], [622, 429], [697, 433], [791, 549]], np.int32)
##        DY_ROAD_CLR_ROI_R = roi(hsv, [DY_ROAD_CLR_VERTICES_R])

##        DY_ROAD_CLR_ROI = cv2.bitwise_or(DY_ROAD_CLR_ROI_L, DY_ROAD_CLR_ROI_R)

##        new_road_colors(DY_ROAD_CLR_VERTICES_L, DY_ROAD_CLR_VERTICES_R, hsv)
        
        LEFT =      color_road1(hsv, 27,  415, 0,   358, 123, 319, 370, 289, 303, 319, 237, 327)
        RIGHT =     color_road1(hsv, 452, 289, 676, 308, 797, 334, 791, 405, 529, 318, 490, 317)
        FORWARD =   color_road0(hsv, 303, 319, 370, 289, 452, 289, 490, 317)

        L = cv2.cvtColor(LEFT,   cv2.COLOR_BGR2GRAY)
        R = cv2.cvtColor(RIGHT,   cv2.COLOR_BGR2GRAY)
        F = cv2.cvtColor(FORWARD,   cv2.COLOR_BGR2GRAY)
        if not paused:
            if s_time >= 1:
                Flag, LR_time = LRF_drive(L, R, F, Flag, LR_time, 2.0)
                s_time = time.time()

        LRF = cv2.bitwise_or(LEFT, RIGHT)
        LRF = cv2.bitwise_or(LRF, FORWARD)

##            LANES = process_img(frame, vertices_frame, hsv)

        cv2.imshow('LRF', LRF)

##            frame_gray =    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##            LRF_gray   =    cv2.cvtColor(LRF,   cv2.COLOR_BGR2GRAY)
##
##            MIN_SIZE = 120
##            MAX_SIZE = 160
##            
##            frame_resize = cv2.resize(frame_gray,   (MAX_SIZE, MIN_SIZE))
##            LANES_resize = cv2.resize(LANES,        (MAX_SIZE, MIN_SIZE))
##            LRF_resize   = cv2.resize(LRF_gray,     (MAX_SIZE, MIN_SIZE))
##
##            if printed:
##                print('frame_resize: ',   frame_resize.shape, 'type: ', type(frame_resize[0][0]))
##                print('LANES_resize: ',   LANES_resize.shape, 'type: ', type(LANES_resize[0][0]))
##                print('LRF_resize: ',     LRF_resize.shape, 'type: ', type(LRF_resize[0][0]))
##                printed = False
##            
##            cv2.imshow('FRAME_resize',  frame_resize)
##            cv2.imshow('LANES_resize',  LANES_resize)
##            cv2.imshow('LRF_resize',    LRF_resize)

            
            
##            keyswad = key_check()
##            output = keys_to_output(keyswad)
##
##            frame_training_data.append( [frame_resize,  output])
##            lane_training_data.append(  [LANES_resize,  output])
##            LRF_training_data.append(   [LRF_resize,    output])
##
####            if len(frame_training_data) % 500 == 0:
##                print(len(frame_training_data))
##                np.save(frame_file_name, frame_training_data)
##                g += 1
##            
##            if len(lane_training_data) % 500 == 0:
##                print(len(lane_training_data))
##                np.save(lane_file_name, lane_training_data)
##                g += 1
##            
##            if len(LRF_training_data) % 500 == 0:
##                print(len(LRF_training_data))
##                np.save(LRF_file_name, LRF_training_data)
##                g += 1
##
##            if g == 3:
##                j = True
##                i += 1
##                print('NEXT FILE! i: ', i)
##
##            if i > 10:
##                print('FILE ENDS!')
##                cv2.destroyAllWindows()
##                break
      
        if cv2.waitKey(25) & ('T' in key_check()):
            if paused:
                paused = False
                print('START!!!')
                time.sleep(1)
            else:
                slow_ya_roll()
                print('PAUSE!!!')
                paused = True
                
        if cv2.waitKey(25) & ('Q' in key_check()):
            print(    'FILE ENDS!')
            cv2.destroyAllWindows()
            break

main()
