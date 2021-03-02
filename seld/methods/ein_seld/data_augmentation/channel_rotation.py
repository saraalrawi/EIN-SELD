import random
import numpy as np
import logging
import torch


rotation_pattern = 15

def apply_channel_rotation_foa(x):

    pattern = random.randrange(rotation_pattern)
    x_aug = torch.zeros(x.size())
    w = x[:, 0]
    y = x[:, 1]
    z = x[:, 2]
    x = x[:, 3]

    # w channel never change
    x_aug[:, 0] = w

    # the 15 pattern rotations and reflection
    if (pattern == 0):
            x_aug[:, 1] = -x
            x_aug[:, 3] = y
            x_aug[:, 2] = z
    elif (pattern == 1):
            x_aug[:, 1] = -x
            x_aug[:, 3] = y
            x_aug[:, 2] = -z

    elif (pattern == 2):
            x_aug[:, 1] = y
            x_aug[:, 3] = x
            x_aug[:, 2] = -z

    elif (pattern == 3):
            x_aug[:, 1] = x
            x_aug[:, 3] = -y
            x_aug[:, 2] = z
    elif (pattern == 4):
            x_aug[:, 1] = x
            x_aug[:, 3] = -y
            x_aug[:, 2] = -z

    elif (pattern == 5):
            x_aug[:, 1] = x
            x_aug[:, 3] = -y
            x_aug[:, 2] = z
    elif (pattern == 6):
            x_aug[:, 1] = -x
            x_aug[:, 3] = -y
            x_aug[:, 2] = -z

    elif (pattern == 7):
            x_aug[:, 1] = -y
            x_aug[:, 3] = x
            x_aug[:, 2] = z

    elif (pattern == 8):
            x_aug[:, 1] = -y
            x_aug[:, 3] = x
            x_aug[:, 2] = -z

    elif (pattern == 9):
            x_aug[:, 1] = x
            x_aug[:, 3] = y
            x_aug[:, 2] = z

    elif (pattern == 10):
            x_aug[:, 1] = x
            x_aug[:, 3] = y
            x_aug[:, 2] = -z

    elif (pattern == 11):
            x_aug[:, 1] = y
            x_aug[:, 3] = -x
            x_aug[:, 2] = z

    elif (pattern == 12):
            x_aug[:, 1] = y
            x_aug[:, 3] = -x
            x_aug[:, 2] = -z

    elif (pattern == 13):
            x_aug[:, 1] = -x
            x_aug[:, 3] = -y
            x_aug[:, 2] = z

    elif (pattern == 14):
            x_aug[:, 1] = -x
            x_aug[:, 3] = -y
            x_aug[:, 2] = -z

    else:
        print("Wrong pattern selection")
    return x_aug.cuda(), pattern


def label_rotation_foa(label, pattern):
    """

      """
    label_aug = torch.zeros(label.size())

    if (pattern == 0):
            for n in range(len(label)):
                label_aug[n, 0, 0] = label[n, 0, 1]
                label_aug[n, 0, 1] = - label[n, 0 , 0]
                label_aug[n, 0, 2] = label[n, 0 , 2]

                label_aug[n, 1, 0] = label[n, 1, 1]
                label_aug[n, 1, 1] = - label[n, 1, 0]
                label_aug[n, 1, 2] = label[n, 1, 2]
    elif (pattern == 1):

            for n in range(len(label)):
                label_aug[n, 0, 0] = label[n, 0, 1]
                label_aug[n, 0, 1] = - label[n, 0, 0]
                label_aug[n,0, 2] = - label[n, 0, 2]

                label_aug[n, 1, 0] = label[n, 1, 1]
                label_aug[n, 1, 1] = - label[n, 1, 0]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 2):

            for n in range(len(label)):

                label_aug[n,0, 0] = label[n,0 , 0]
                label_aug[n, 0 ,1] = label[n,0 ,1]
                label_aug[n,0, 2] = - label[n,0 , 2]

                label_aug[n, 1, 0] = label[n, 1, 0]
                label_aug[n, 1, 1] = label[n, 1, 1]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 3):

            for n in range(len(label)):

                label_aug[n,0, 0] = - label[n,0,1]
                label_aug[n,0, 1] = label[n,0,0]
                label_aug[n,0, 2] = label[n,0, 2]

                label_aug[n, 1, 0] = - label[n, 1, 1]
                label_aug[n, 1, 1] = label[n, 1, 0]
                label_aug[n, 1, 2] = label[n, 1, 2]

    elif (pattern == 4):

        for n in range(len(label)):

                label_aug[n, 0, 0] = - label[n, 0, 1]
                label_aug[n, 0, 1] = label[n, 0, 0]
                label_aug[n, 0, 2] = - label[n, 0, 2]

                label_aug[n, 1, 0] = - label[n, 1, 1]
                label_aug[n, 1, 1] = label[n, 1, 0]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 5):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = - label[n, 0 ,1]
                label_aug[n, 0 ,1] = label[n, 0 ,0]
                label_aug[n, 0 ,2] = label[n, 0 ,2]

                label_aug[n, 1, 0] = - label[n, 1, 1]
                label_aug[n, 1, 1] = label[n, 1, 0]
                label_aug[n, 1, 2] = label[n, 1, 2]

    elif (pattern == 6):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = - label[n, 0 ,1]
                label_aug[n, 0  ,1] = - label[n, 0 ,0]
                label_aug[n, 0  ,2] = - label[n, 0 ,2]

                label_aug[n, 1, 0] = - label[n, 1, 1]
                label_aug[n, 1, 1] = - label[n, 1, 0]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 7):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = label[n, 0 ,0]
                label_aug[n, 0 ,1] = - label[n, 0 ,1]
                label_aug[n, 0 ,2] = label[n, 0 ,2]

                label_aug[n, 1, 0] = label[n, 1, 0]
                label_aug[n, 1, 1] = - label[n, 1, 1]
                label_aug[n, 1, 2] = label[n, 1, 2]

    elif (pattern == 8):

            for n in range(len(label)):

                label_aug[n, 0,0] = label[n, 0, 0]
                label_aug[n, 0,1] = - label[n, 0,1]
                label_aug[n, 0,2] = - label[n, 0,2]

                label_aug[n, 1, 0] = label[n, 1, 0]
                label_aug[n, 1, 1] = - label[n, 1, 1]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 9):

            for n in range(len(label)):

                label_aug[n, 0, 0] = label[n, 0 ,1]
                label_aug[n, 0, 1] = label[n, 0 ,0]
                label_aug[n, 0, 2] = label[n, 0 ,2]

                label_aug[n, 1, 0] = label[n, 1, 1]
                label_aug[n, 1, 1] = label[n, 1, 0]
                label_aug[n, 1, 2] = label[n, 1, 2]

    elif (pattern == 10):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = label[n, 0 ,1]
                label_aug[n, 0 ,1] = label[n, 0 ,0]
                label_aug[n, 0 ,2] = - label[n, 0 ,2]

                label_aug[n, 1, 0] = label[n, 1, 1]
                label_aug[n, 1, 1] = label[n, 1, 0]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 11):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = - label[n, 0 ,0]
                label_aug[n, 0 ,1] = label[n, 0 ,1]
                label_aug[n, 0 ,2] = label[n, 0 ,2]

                label_aug[n, 1, 0] = - label[n, 1, 0]
                label_aug[n, 1, 1] = label[n, 1, 1]
                label_aug[n, 1, 2] = label[n, 1, 2]

    elif (pattern == 12):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = - label[n, 0, 0]
                label_aug[n, 0 ,1] = label[n, 0 ,1]
                label_aug[n, 0 ,2] = - label[n, 0 ,2]

                label_aug[n, 1, 0] = - label[n, 1, 0]
                label_aug[n, 1, 1] = label[n, 1, 1]
                label_aug[n, 1, 2] = - label[n, 1, 2]

    elif (pattern == 13):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = - label[n, 0 ,1]
                label_aug[n, 0 ,1] = - label[n, 0 ,0]
                label_aug[n, 0 ,2] = label[n, 0 ,2]

                label_aug[n, 1, 0] = - label[n, 1, 1]
                label_aug[n, 1, 1] = - label[n, 1, 0]
                label_aug[n, 1, 2] = label[n, 1, 2]

    elif (pattern == 14):

            for n in range(len(label)):

                label_aug[n, 0 ,0] = - label[n, 0 ,1]
                label_aug[n, 0 ,1] = - label[n, 0 ,0]
                label_aug[n, 0 ,2] = - label[n, 0 ,2]

                label_aug[n, 1, 0] = - label[n, 1, 1]
                label_aug[n, 1, 1] = - label[n, 1, 0]
                label_aug[n, 1, 2] = - label[n, 1, 2]
    else:
        print("Wrong pattern selection")
    return label_aug.cuda()
#list_labels
def apply_data_channel_rotation(dataset_type,x):
    if dataset_type == 'foa':
        #print("Data augmentation pattern {}".format(pattern))
        x_aug, pattern = apply_channel_rotation_foa(x)
        return x_aug, pattern
    '''
    elif dataset_type == 'mic':
        apply_channel_rotation_mic(list_audio_data)
    else:
        logging.info('Choose correct dataset type for channel rotation data augmentation.')
    '''
def apply_label_channel_rotation(dataset_type,label, pattern):
    if dataset_type == 'foa':
        label_aug = label_rotation_foa( label, pattern)
        return label_aug
    elif dataset_type == 'mic':
        pass
    else:
        logging.info('Choose correct dataset type for channel rotation data augmentation.')
