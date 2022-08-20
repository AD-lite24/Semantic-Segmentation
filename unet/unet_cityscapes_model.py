from tensorflow import keras
import h5py


f = h5py.File('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/unet/unet.h5', mode='r')
x = f.attrs.get()
print(f.attrs.get('model_config'))
# model = keras.models.load_model('/Users/adityadandwate/Desktop/Projects/Proper/sem_seg/unet/unet.h5')
