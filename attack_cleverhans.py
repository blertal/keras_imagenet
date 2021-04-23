from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import CarliniWagnerL2 
from cleverhans.attacks import DeepFool
from keras import backend as K
import numpy as np

# Cleverhans attacks, used as they are faster than art toolbox attacks

# CWL2 attack
def get_CWL2_adversarial(targeted, xs, y_target, classifier, batch_size, cwl2_confidence):

    #print(xs.shape, y_target.shape)
    #exit()
    ATTACK_BATCH = batch_size
    samples_range  = int(xs.shape[0]/ATTACK_BATCH)

    wrap = KerasModelWrapper(classifier)
    attack = CarliniWagnerL2(wrap, sess=K.get_session())
    fgsm_params = {'confidence':cwl2_confidence,
                   'max_iterations':1000,
                   'binary_search_steps':9,
                   'initial_const':1,
                   'clip_min':-5,'clip_max':5,
                   'batch_size':ATTACK_BATCH}

    if targeted:
        y_target = np.expand_dims(y_target, axis=1)

        attack_xs  = attack.generate_np(xs[:ATTACK_BATCH,:,:,:], y_target=y_target[:ATTACK_BATCH], **fgsm_params)
        for ii in range(1,samples_range):
            print('iter', ii)
            new_attack_batch = attack.generate_np(xs[ii*ATTACK_BATCH:(ii+1)*ATTACK_BATCH,:,:,:], 
                                                  y_target=y_target[ii*ATTACK_BATCH:(ii+1)*ATTACK_BATCH],
                                                  **fgsm_params)
            attack_xs = np.concatenate((attack_xs, new_attack_batch), axis=0)
    else:
        attack_xs  = attack.generate_np(xs[:ATTACK_BATCH,:,:,:], **fgsm_params)
        for ii in range(1,samples_range):
            print('iter', ii)
            new_attack_batch = attack.generate_np(xs[ii*ATTACK_BATCH:(ii+1)*ATTACK_BATCH,:,:,:],**fgsm_params)
            attack_xs = np.concatenate((attack_xs, new_attack_batch), axis=0)
    return attack_xs

# DeepFool attack
def get_DeepFool_adversarial(targeted, xs, classifier, batch_size):

    # Targeted DeepFool attack not possible
    if targeted:
        print('DeepFool attack cannot be targeted.')
        exit()

    ATTACK_BATCH = batch_size
    samples_range  = int(xs.shape[0]/ATTACK_BATCH)

    wrap = KerasModelWrapper(classifier)
    attack = DeepFool(wrap, sess=K.get_session())
    fgsm_params = { 'overshoot':0.02, 'max_iter':50,
                    'nb_candidate':2, 'clip_min':-5,
                    'clip_max':5}
    
    attack_xs  = attack.generate_np(xs[:ATTACK_BATCH,:,:,:], **fgsm_params)
    for ii in range(1,samples_range):
        print('ITER', ii)
        new_attack_batch = attack.generate_np(xs[ii*ATTACK_BATCH:(ii+1)*ATTACK_BATCH,:,:,:], **fgsm_params)
        attack_xs = np.concatenate((attack_xs, new_attack_batch), axis=0)
    return attack_xs

