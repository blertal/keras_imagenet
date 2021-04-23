import numpy as np
from keras.utils.np_utils import to_categorical
from art.classifiers import KerasClassifier
from attack_art import *
from attack_cleverhans import *

DEFAULT_BATCH_SIZE = 100

## Get accuracy of predictions
#def get_accuracy(dataset, grndTruth, discriminator, num_classes=10):
#
#    # Get predicted probabilities
#    pred = discriminator.predict(dataset)
#
#    # Add up real and designated probabilitites
#    for jj in range(num_classes):
#        pred[:,jj] += pred[:,jj+num_classes]
#
#    # Keep only the above addition result
#    pred = pred[:, 0:num_classes]
#
#    pred = np.argmax(pred, axis=1)
#    pred = np.reshape(pred, (pred.shape[0], 1))
#
#    grndTruth = grndTruth.astype(int)
#    correct = np.count_nonzero(pred==grndTruth)
#    print('CORRECT',correct)
#    acc =  correct*1.0/pred.shape[0]
#    print('Curr accuracy', acc)
#
#    return acc


def get_targeted_ys(ys):
    
    dict_lbls = {}

    LABEL_COUNT = 1000
    for ii in range(LABEL_COUNT):
        dict_lbls[ii] = list(range(LABEL_COUNT))
        dict_lbls[ii].remove(ii)

    targeted_ys = np.zeros_like(ys)
    for ii in range(ys.shape[0]):
        # Choose randomly among the other labels
        ch = dict_lbls[ys[ii]]
        targeted_ys[ii] = np.random.choice(ch,1)

    return targeted_ys

# dataset is needed only by PGD attack
# fgsm_epsilon is only for FGSM attack
# cwl2_confidence is only for CWL2 attack
def run_attack(targeted, attack_name, classifier, samples, y_samples, batch_size=DEFAULT_BATCH_SIZE, dataset='mnist', fgsm_epsilon=0.1, cwl2_confidence=0):

    x_test_new = None

    targeted_y_samples = None
    if targeted:
        targeted_y_samples = get_targeted_ys(y_samples)

    #-------------------
    # CleverHans attacks
    if attack_name == 'DeepFool':
        x_test_new  = get_DeepFool_adversarial(targeted, samples, classifier, batch_size)
    elif attack_name == 'CarliniL2Method':
        x_test_new  = get_CWL2_adversarial(targeted, samples, targeted_y_samples, classifier, 
                                           batch_size, cwl2_confidence)
    #-----------------
    else:# ART attacks
        classifier_copy = KerasClassifier(model=classifier, clip_values=(0.0,1.0), use_logits=False)
        x_test_new   = get_adversarial(targeted, attack_name, classifier_copy, samples, targeted_y_samples, 
                                       batch_size, dataset, fgsm_epsilon, cwl2_confidence)

    return x_test_new

