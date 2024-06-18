import tensorflow as tf
import matplotlib.pyplot as plt
import os, sys
if len(sys.argv)!=2:
    print('usage python make_plot.py model')
    sys.exit(3)
history = tf.keras.models.load_model(sys.argv[1])

# Plot training & validation accuracy values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.plot(epochs, loss, 'r--', label='Training loss')
plt.plot(epochs, val_loss, 'b--', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig('training_acc_loss.png')

def load_model(model_path):
    return tf.keras.models.load_model(model_path)