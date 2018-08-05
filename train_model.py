import numpy as np
from alexnet import alexnet

###########################################################################

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 10
MODEL_NAME = 'pygta5-car-fast-{}-{}-{}-epochs-300K-data.model'.format(LR, 'alexnetv2',EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

hm_data = 22
for i in range(EPOCHS):
    for i in range(1, hm_data + 1):
        train_data = np.load('training_data-{}-balanced.npy'.format(i))

        train = train_data[:-100]
        test = train_data[-100:]

        X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
        Y = [i[1] for i in train]

        test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
        test_y = [i[1] for i in test]

        print('lets go fit')
        model.fit({'input': X}, {'targets': Y}, n_epoch=1, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
        print('fit ends')
        
        model.save(MODEL_NAME)

###########################################################################

##MODEL_NAME_1 = 'pygta5-car-LANES-{}-{}-{}-epochs'.format(LR, 'alexnetv2', EPOCHS)
##
##model_1 = alexnet(WITH, HEIGHT, LR)
##
##train_data_1 = np.load('training_lane_v2.npy')
##
##train_1 = train_data_1[:-500]
##test_1 = train_data_1[:-500:]
##
##X_1 = np.array([i[0] for i in train_1]).reshape(-1, WITH, HEIGHT, 1)
##Y_1 = [i[1] for i in train_1]
##
##test_x_1 = np.array([i[0] for i in test_1]).reshape(-1, WITH, HEIGHT, 1)
##test_y_1 = [i[1] for i in test_1]
##
##print('X_1 shape:', X_1.shape)
##
##model_1.fit({'input': X_1}, {'targets': Y_1}, n_epoch = EPOCHS,
##          validation_set = ({'input': test_x_1}, {'targets': test_y_1}),
##          snapshot_step = 500, show_metric = True, run_id = MODEL_NAME_1)
##
##model_1.save(MODEL_NAME_1)

###########################################################################

##MODEL_NAME_2 = 'pygta5-car-LRF-{}-{}-{}-epochs'.format(LR, 'alexnetv2', EPOCHS)
##
##model_2 = alexnet(WITH, HEIGHT, LR)
##
##train_data_2 = np.load('training_LRF_v2.npy')
##
##train_2 = train_data_2[:-500]
##test_2 = train_data_2[:-500:]
##
##X_2 =         np.array([i[0] for i in train_2]).reshape(-1, WITH, HEIGHT, 1)
##Y_2 =                  [i[1] for i in train_2]
##
##test_x_2 =    np.array([i[0] for i in test_2]).reshape(-1, WITH, HEIGHT, 1)
##test_y_2 =             [i[1] for i in test_2]
##
##print('X_2 shape:', X_2.shape)
##
##model_2.fit({'input': X_2}, {'targets': Y_2}, n_epoch = EPOCHS,
##          validation_set = ({'input': test_x_2}, {'targets': test_y_2}),
##          snapshot_step = 500, show_metric = True, run_id = MODEL_NAME_2)
##
##model_2.save(MODEL_NAME_2)
