import numpy as np
#from art.attacks import *
from art.attacks import FastGradientMethod
from art.attacks import CarliniL2Method
from art.attacks import DeepFool
from art.attacks import CarliniLInfMethod
from art.attacks import ProjectedGradientDescent
from art.attacks import UniversalPerturbation
from art.attacks import ZooAttack
#from art.classifiers import KerasClassifier

# IBM ART toolbox attacks

def get_adversarial(targeted, attack_name, classifier, xs, target_ys, batch_size, dataset, fgsm_epsilon=0, cwl2_confidence=0):


    # The attack
    attack = ''
    samples_range = xs.shape[0]

    #======================================
    if attack_name == 'FastGradientMethod':
        # norm=np.inf, eps=.3, eps_step=0.1, targeted=False, num_random_init=0, batch_size=1,minimal=False
        attack = FastGradientMethod(classifier=classifier, 
                targeted=targeted, eps=fgsm_epsilon, batch_size=batch_size)
    #=====================================
    elif attack_name == 'CarliniLInfMethod':
        # confidence=0.0, targeted=False, learning_rate=0.01, max_iter=10, max_halving=5,
        #max_doubling=5, eps=0.3, batch_size=128
        attack = CarliniLInfMethod(classifier=classifier, max_iter=1000, 
                targeted=targeted, batch_size=batch_size)
    #-------------------------------
    elif attack_name == 'UniversalPerturbation':
        # attacker='deepfool', attacker_params=None, delta=0.2, 
        # max_iter=20, eps=10.0, norm=np.inf

        if targeted:
            print('UniversalPerturbation attack cannot be targeted.')
            exit()
        attack = UniversalPerturbation(classifier=classifier, max_iter=5)

    #==============================================
    elif attack_name == 'ProjectedGradientDescent':
        # norm=np.inf, eps=.3, eps_step=0.1, max_iter=100,
        # targeted=False, num_random_init=0, batch_size=1
        if dataset == 'mnist':
            attack = ProjectedGradientDescent(classifier=classifier, targeted=targeted, 
                    norm=1, 
                    eps=.3, eps_step=0.01, num_random_init=0, 
                                              max_iter=40, batch_size=batch_size)
        else:
            attack = ProjectedGradientDescent(classifier=classifier, targeted=targeted, 
                         norm=1,
                         eps=8.0, eps_step=2.0, num_random_init=0,
                                              max_iter=7, batch_size=batch_size)

    if targeted:
        # Generate the adversarial samples in steps
        adv = attack.generate(xs[0:batch_size,:,:,:] , y=target_ys[0:batch_size])###################
        last_ii = 0
        for ii in range(batch_size,samples_range-batch_size,batch_size):
            print(ii)
            adv_samples = attack.generate(xs[ii:ii+batch_size,:,:,:], y=target_ys[ii:ii+batch_size])####################
            adv = np.concatenate((adv, adv_samples), axis=0)
            last_ii = ii

        # The rest of the samples
        if last_ii + batch_size < xs.shape[0]:
            last_samples = xs[last_ii+batch_size:,:,:,:]
            adv_samples = attack.generate(last_samples, y=target_ys[last_ii+batch_size:])################
            adv = np.concatenate((adv, adv_samples), axis=0)
    else:
        # Generate the adversarial samples in steps
        adv = attack.generate(xs[0:batch_size,:,:,:])###################
        last_ii = 0
        for ii in range(batch_size,samples_range-batch_size,batch_size):
            print(ii)
            adv_samples = attack.generate(xs[ii:ii+batch_size,:,:,:])####################
            adv = np.concatenate((adv, adv_samples), axis=0)
            last_ii = ii

        # The rest of the samples
        if last_ii + batch_size < xs.shape[0]:
            last_samples = xs[last_ii+batch_size:,:,:,:]
            adv_samples = attack.generate(last_samples)################
            adv = np.concatenate((adv, adv_samples), axis=0)

    adv = np.asarray(adv)
    return adv

