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
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--inv_model_file', type=str,
                        help='a saved model (.h5) file')
    args = parser.parse_args()
    config_keras_backend()
    if not args.inv_model_file.endswith('.h5'):
        sys.exit('model_file is not a .h5')
    inv_model = tf.keras.models.load_model(
        args.inv_model_file,
        compile=False,
        custom_objects={'AdamW': AdamW})
    inv_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    ds_validation = get_dataset(
        args.dataset_dir, 'validation', args.batch_size)


    ## VGG
    vgg_model = VGG19(include_top=True, weights='imagenet', classes=1000)
    vgg_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    # InceptionV3
    inception_model = InceptionV3(include_top=True, weights='imagenet', classes=1000)
    inception_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    ## ResNet
    resnet_model = ResNet50(include_top=True, weights='imagenet', classes=1000)
    resnet_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])


    # Process batches
    iteration = 0
    sum1 = 0
    sum2 = 0
    for images, labels in tfds.as_numpy(ds_validation):

        if iteration < 0:
            print('continuing')
            iteration += 1
            continue
        if iteration == 500:
            exit()

        labels = np.argmax(labels, axis=1)

        adv_imgs = run_attack(True, 'CarliniL2Method', inception_model, images, labels, batch_size=args.batch_size, dataset='cifar', fgsm_epsilon=0.3, cwl2_confidence=0)
        #adv_imgs = run_attack(False, 'DeepFool', inception_model, images, labels, batch_size=args.batch_size, dataset='cifar', fgsm_epsilon=0.3, cwl2_confidence=0)
        #adv_imgs = run_attack(False, 'FastGradientMethod', inception_model, images, labels, batch_size=args.batch_size, dataset='cifar', fgsm_epsilon=0.1, cwl2_confidence=0)

        ## VGG ################################################

        #img *= (2.0/255)  # normalize to: 0.0~2.0
        #img -= 1.0        # subtract mean to make it: -1.0~1.0
        #img = np.expand_dims(img, axis=0)

        vgg_imgs = []
        resnet_imgs = []
        inc_imgs = []
        flip_imgs = []
        inv_imgs = []
        adv_vgg_imgs = []
        adv_resnet_imgs = []
        adv_inc_imgs = []
        adv_flip_imgs = []
        adv_inv_imgs = []
        for ii in range(images.shape[0]):
            img = copy.deepcopy(images[ii,:,:,:])
            img += 1.0
            #img /= (2.0/255)
            img *= (255.0/2.0)

            ## VGG
            vgg_img = copy.deepcopy(img)
            vgg_img = cv2.resize(vgg_img, (224, 224))
            vgg_img = vgg_preprocess_input(vgg_img)
            vgg_imgs.append(vgg_img)
           
            ## Resnet
            resnet_img = copy.deepcopy(img)
            resnet_img = cv2.resize(resnet_img, (224, 224))
            resnet_img = resnet_preprocess_input(resnet_img)
            resnet_imgs.append(resnet_img)

            ## InceptionV3
            inc_img = copy.deepcopy(img)
            inc_img = cv2.resize(inc_img, (299, 299))
            inc_img = inception_preprocess_input(inc_img)
            inc_imgs.append(inc_img)

            ## Flipped
            #flip_img = copy.deepcopy(img)
            #flip_img = cv2.resize(flip_img, (299, 299))
            #flip_img = cv2.flip(flip_img, 1)
            #flip_img = inception_preprocess_input(flip_img)
            #flip_imgs.append(flip_img)
            flip_img = copy.deepcopy(images[ii,:,:,:])
            flip_img = cv2.flip(flip_img, 1)
            flip_imgs.append(flip_img)

            ## Inverse
            inv_img = copy.deepcopy(images[ii,:,:,:])#########
            inv_img += 1.0
            inv_img /= 2.0
            inv_img = 1 - inv_img
            inv_img *= 255.0
            inv_img = cv2.resize(inv_img, (299, 299))
            inv_img = inception_preprocess_input(inv_img)
            inv_imgs.append(inv_img)


            #==========================================
            # ADVERSARIAL ---------------
            adv_img = copy.deepcopy(adv_imgs[ii,:,:,:])
            adv_img += 1.0
            #adv_img /= (2.0/255)
            adv_img *= (255.0/2.0)

            # VGG
            adv_vgg_img = copy.deepcopy(adv_img)
            adv_vgg_img = cv2.resize(adv_vgg_img, (224, 224))
            adv_vgg_img = vgg_preprocess_input(adv_vgg_img)
            adv_vgg_imgs.append(adv_vgg_img)

            # Resnet
            adv_resnet_img = copy.deepcopy(adv_img)
            adv_resnet_img = cv2.resize(adv_resnet_img, (224, 224))
            adv_resnet_img = resnet_preprocess_input(adv_resnet_img)
            adv_resnet_imgs.append(adv_resnet_img)

            # InceptionV3
            adv_inc_img = copy.deepcopy(adv_img)
            adv_inc_img = cv2.resize(adv_inc_img, (299, 299))
            adv_inc_img = inception_preprocess_input(adv_inc_img)
            adv_inc_imgs.append(adv_inc_img)

            ## Flipped
            #adv_flip_img = copy.deepcopy(img)
            #adv_flip_img = cv2.resize(adv_flip_img, (299, 299))
            #adv_flip_img = cv2.flip(adv_flip_img, 1)
            #adv_flip_img = inception_preprocess_input(adv_flip_img)
            #adv_flip_imgs.append(adv_flip_img)
            adv_flip_img = copy.deepcopy(adv_imgs[ii,:,:,:])
            adv_flip_img = cv2.flip(adv_flip_img, 1)
            adv_flip_imgs.append(adv_flip_img)

            ## Inverse
            ##test on inverse Inceptionv3
            adv_inv_img = copy.deepcopy(adv_imgs[ii,:,:,:])#########
            adv_inv_img += 1.0
            adv_inv_img /= 2.0
            adv_inv_img = 1 - adv_inv_img
            adv_inv_img *= 255.0
            adv_inv_img = cv2.resize(adv_inv_img, (299, 299))
            adv_inv_img = inception_preprocess_input(adv_inv_img)
            adv_inv_imgs.append(adv_inv_img)

            # Horizontal Flipping
            # test on Resnet


        vgg_imgs = np.asarray(vgg_imgs)
        resnet_imgs = np.asarray(resnet_imgs)
        inc_imgs = np.asarray(inc_imgs)
        flip_imgs = np.asarray(flip_imgs)
        inv_imgs = np.asarray(inv_imgs)

        adv_vgg_imgs = np.asarray(adv_vgg_imgs)
        adv_resnet_imgs = np.asarray(adv_resnet_imgs)
        adv_inc_imgs = np.asarray(adv_inc_imgs)
        adv_flip_imgs = np.asarray(adv_flip_imgs)
        adv_inv_imgs = np.asarray(adv_inv_imgs)


        # Default ResNet accuracy
#        print('Iteration', iteration)
#        print('LABELS              :', labels)

##        results = resnet_model.predict(x=resnet_imgs, verbose=0)
##        results = np.argmax(results, axis=1)
##        print('RESNET     default  :', results)
#
##        results = vgg_model.predict(x=vgg_imgs, verbose=0)
##        results = np.argmax(results, axis=1)
##        print('VGG        default  :', results)
#
#        results = inception_model.predict(x=inc_imgs, verbose=0)
#        results = np.argmax(results, axis=1)
#        print('Inception  default  :', results)
#
#        results = inv_model.predict(x=inv_imgs, verbose=0)
#        results = np.argmax(results, axis=1)
#        print('Inv-incep  default  :', results)
#
#        print('------------------------------------------------')
#        
##        results = resnet_model.predict(x=adv_resnet_imgs, verbose=0)
##        results = np.argmax(results, axis=1)
##        print('RESNET     adv      :', results)
#
##        results = vgg_model.predict(x=adv_vgg_imgs, verbose=0)
##        results = np.argmax(results, axis=1)
##        print('VGG        adv      :', results)
#
#        results = inception_model.predict(x=adv_inc_imgs, verbose=0)
#        results = np.argmax(results, axis=1)
#        print('Inception  adv      :', results)
#
#        results = inv_model.predict(x=adv_inv_imgs, verbose=0)
#        results = np.argmax(results, axis=1)
#        print('Inv-incep  adv      :', results)
#
#        print('=====================================================')



        # Default ResNet accuracy
        _, results1 = resnet_model.evaluate(x=resnet_imgs, y=labels, verbose=0)
        #print('RESNET     default  test loss, test acc:', results)

        _, results2 = vgg_model.evaluate(x=vgg_imgs, y=labels, verbose=0)
        #print('VGG        default  test loss, test acc:', results)

        _, results3 = inception_model.evaluate(x=inc_imgs, y=labels, verbose=0)
        #print('Inception  default  test loss, test acc:', results)

        _, results4 = inception_model.evaluate(x=flip_imgs, y=labels, verbose=0)
        #print('Inception  default  test loss, test acc:', results)

        _, results5 = inv_model.evaluate(x=inv_imgs, y=labels, verbose=0)
        #print('Inception  default  test loss, test acc:', results)

#        print('-----------------------------------------------------')

        _, results6 = resnet_model.evaluate(x=adv_resnet_imgs, y=labels, verbose=0)
#        print('RESNET     adv  test loss, test acc:', results)

        _, results7 = vgg_model.evaluate(x=adv_vgg_imgs, y=labels, verbose=0)
#        print('VGG        adv  test loss, test acc:', results)

        _, results8 = inception_model.evaluate(x=adv_inc_imgs, y=labels, verbose=0)
        #print('Inception  adv  test loss, test acc:', results)

        _, results9 = inception_model.evaluate(x=adv_flip_imgs, y=labels, verbose=0)
        #print('Inception  adv  test loss, test acc:', results)

        _, results10 = inv_model.evaluate(x=adv_inv_imgs, y=labels, verbose=0)
        #print('Inception  adv  test loss, test acc:', results)

        print(iteration)
        print(results1, results6)
        print(results2, results7)
        print(results3, results8)
        print(results4, results9)
        print(results5, results10)

        #with open("output_cw40_100.txt", "a") as myfile:
            #myfile.write(str(results1) + ' ' + str(results2) +  ' ' +  str(results3) + ' ' + str(results4) + ' ' +  str(results5) + ' ' + str(results6) + ' ' +  str(results7) + ' ' + str(results8) + ' ' +  str(results9) + ' ' + str(results10) +  '\n'     )


        iteration += 1


        #exit()

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
