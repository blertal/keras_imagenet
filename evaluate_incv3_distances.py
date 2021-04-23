"""evaluate.py

This script is used to evalute trained ImageNet models.
"""


import sys
import argparse

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

from config import config
from utils.utils import config_keras_backend, clear_keras_session
from utils.dataset import get_dataset
from models.adamw import AdamW

from keras.utils import to_categorical
from methods import run_attack
#from tensorflow.keras.applications import InceptionV3
#from tensorflow.keras.applications import VGG19
#from tensorflow.keras.applications import ResNet152V2
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
#from keras.applications.resnet101 import ResNet101
from keras.applications.vgg19 import VGG19, decode_predictions
from keras.applications.vgg19 import preprocess_input as vgg_preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_preprocess_input

from methods import get_accuracy, run_attack
#from tf.keras.preprocessing.image import ImageDataGenerator
import cv2
import copy
import scipy

DESCRIPTION = """For example:
$ python3 evaluate.py --dataset_dir ${HOME}/data/ILSVRC2012/tfrecords \
                      --batch_size  64 \
                      saves/mobilenet_v2-model-final.h5

python3 evaluate_resnet_all.py --dataset_dir /l/IMAGENET_ORIGINAL/train/imagenet_tfrecord --inv_model_file /l/keras_imagenet-master/saves/inception_v3-ckpt-030_orig.h5

"""


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--dataset_dir', type=str,
                        default=config.DEFAULT_DATASET_DIR)
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()
    config_keras_backend()

    ds_validation = get_dataset(
        args.dataset_dir, 'validation', args.batch_size)


    # InceptionV3
    inception_model = InceptionV3(include_top=True, weights='imagenet', classes=1000)
    inception_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


    # Process batches
    iteration = 0
    sum1 = 0
    sum2 = 0
    for images, labels in tfds.as_numpy(ds_validation):

        if iteration < 31:
            print('continuing')
            iteration += 1
            continue
        if iteration == 1000:
            exit()

        labels = np.argmax(labels, axis=1)

        #adv_imgs = run_attack(False, 'CarliniL2Method', inception_model, images, labels, batch_size=5, dataset='cifar', fgsm_epsilon=0.3, cwl2_confidence=40)
        adv_imgs = run_attack(False, 'CarliniLInfMethod', inception_model, images, labels, batch_size=5, dataset='cifar', fgsm_epsilon=0.3, cwl2_confidence=0)
        #adv_imgs = run_attack(False, 'DeepFool', inception_model, images, labels, batch_size=args.batch_size, dataset='cifar', fgsm_epsilon=0.3, cwl2_confidence=0)
        #adv_imgs = run_attack(True, 'FastGradientMethod', inception_model, images, labels, batch_size=args.batch_size, dataset='cifar', fgsm_epsilon=0.1, cwl2_confidence=0)
        #adv_imgs = run_attack(True, 'ProjectedGradientDescent', inception_model, images, labels, batch_size=args.batch_size, dataset='cifar', fgsm_epsilon=0.1, cwl2_confidence=0)
        ## VGG ################################################

        inc_imgs = []
        adv_inc_imgs = []
        for ii in range(images.shape[0]):
            img = copy.deepcopy(images[ii,:,:,:])
            img += 1.0
            #img /= (2.0/255)
            img *= (255.0/2.0)


            ## InceptionV3
            inc_img = copy.deepcopy(img)
            inc_img = cv2.resize(inc_img, (299, 299))
            inc_img = inception_preprocess_input(inc_img)
            inc_imgs.append(inc_img)


            #==========================================
            # ADVERSARIAL ---------------
            adv_img = copy.deepcopy(adv_imgs[ii,:,:,:])
            adv_img += 1.0
            #adv_img /= (2.0/255)
            adv_img *= (255.0/2.0)


            # InceptionV3
            adv_inc_img = copy.deepcopy(adv_img)
            adv_inc_img = cv2.resize(adv_inc_img, (299, 299))
            adv_inc_img = inception_preprocess_input(adv_inc_img)
            adv_inc_imgs.append(adv_inc_img)


        inc_imgs = np.asarray(inc_imgs)

        adv_inc_imgs = np.asarray(adv_inc_imgs)


        # Default ResNet accuracy

#        _, results3 = inception_model.evaluate(x=inc_imgs, y=labels, verbose=0)
#        _, results8 = inception_model.evaluate(x=adv_inc_imgs, y=labels, verbose=0)

        adv_inc_imgs = np.nan_to_num(adv_inc_imgs)
        inc_imgs     = np.nan_to_num(inc_imgs)

        norm_diffs_1   = [  np.linalg.norm(np.subtract(adv_inc_imgs[ii].flatten(),inc_imgs[ii].flatten()),1)  for ii in range(inc_imgs.shape[0])]
        norm_diffs_2   = [  np.linalg.norm(np.subtract(adv_inc_imgs[ii].flatten(),inc_imgs[ii].flatten()),2)  for ii in range(inc_imgs.shape[0])]
        norm_diffs_inf = [  np.linalg.norm(np.subtract(adv_inc_imgs[ii].flatten(),inc_imgs[ii].flatten()),np.inf)  for ii in range(inc_imgs.shape[0])]


        print(iteration)
        print(np.mean(norm_diffs_1), np.mean(norm_diffs_2), np.mean(norm_diffs_inf))

        #with open("distances_cw0_untarg.txt", "a") as myfile:
        #    myfile.write(str(np.mean(norm_diffs_1)) + ' ' + str(np.mean(norm_diffs_2)) +  ' ' +  str(np.mean(norm_diffs_inf)) +  '\n'     )



        iteration += 1

        print(norm_diffs_1)
        #print(adv_inc_imgs[0])
        #print(inc_imgs[0])
        exit()

        #results = resnet_model.evaluate(x=adv_imgs, y=to_categorical(labels, 1000))
        #print('RESNET test loss, test acc:', results)
        #results = vgg_model.evaluate(x=adv_imgs, y=to_categorical(labels, 1000))
        #print('VGG    test loss, test acc:', results)



 #        labels = np.argmax(labels, axis=1)
 #
 #        #results = model.evaluate(
 #        #               x=images, y=to_categorical(labels, 1000))
 #        #print('test loss, test acc:', results)
 #        total = total + images.shape[0]
 #    print(total)
    exit()


    results = resnet_model.evaluate(
        x=ds_validation,
        steps=50000 // args.batch_size)
    print('test loss, test acc:', results)
    clear_keras_session()


if __name__ == '__main__':
    main()
