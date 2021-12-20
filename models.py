import tensorflow as tf
from tensorflow.keras.layers import *
from metrics import CKA


train_data,validation_data = tf.keras.datasets.cifar10.load_data()
train_images=tf.keras.applications.resnet.preprocess_input(train_data[0])
validation_images=tf.keras.applications.resnet.preprocess_input(validation_data[0])

train_labels =tf.keras.utils.to_categorical(train_data[1],10)
validation_labels=tf.keras.utils.to_categorical(validation_data[1],10)

train_data = (train_images,train_labels)

val_data=(validation_images[6000:],validation_labels[6000:])
test_data = (validation_images[:4000],validation_labels[:4000])


def head():
    head = tf.keras.models.Sequential([
            Dense(265,activation='relu'),
            Dense(10,activation='softmax')
    ])
    return head

def resnet50():
    h=head()
    model =tf.keras.applications.resnet.ResNet50(weights='imagenet',pooling='avg',include_top=False,input_shape=(32,32,3))

    return tf.keras.models.Model(model.input,h(model.output))

def resnet101():
    h=head()
    model =tf.keras.applications.resnet.ResNet101(weights='imagenet',pooling='avg',include_top=False,input_shape=(32,32,3))

    return tf.keras.models.Model(model.input,h(model.output))

def resnet152():
    h=head()
    model =tf.keras.applications.resnet.ResNet152(weights='imagenet',pooling='avg',include_top=False,input_shape=(32,32,3))

    return tf.keras.models.Model(model.input,h(model.output))


def get_activations(model):
    activations=tf.keras.Model(model.input,[l.output for l in model.layers])
    return activations


def compare_activations(model1,model2):
    activations1 = get_activations(model1)
    activations2 = get_activations(model2)

    list1=[]
    list2=[]
    for act1 in activations1.outputs:
        for act2 in activations2.outputs:
            list2.append(CKA(act1,act2))
        list1.append(list2)
        list2=[]
    return list1[]


def train(model):

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    cp_tensorboard=tf.keras.callbacks.TensorBoard(
                                    log_dir='logs', update_freq='epoch')

    model.compile(optimizer='adam',loss='categorical_crossentropy')
    model.fit(train_data,
              validation_data=val_data,
              callbacks=[cp_callback,cp_tensorboard]
              batch_size=16
              )
    