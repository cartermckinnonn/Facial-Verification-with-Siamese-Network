# Carter McKinnon August 2022
# This will compile, fit, evaluate, and save the siamese model

from SiameseBuilder import SiameseNetwork
import ProcessingTools
from keras import callbacks
import tensorflow as tf
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

# file paths to csv's of data
validation_set_path = "Data/ValidationSet.csv"
train_set_path = "Data/TrainSet.csv"
test_set_path = "Data/TestSet.csv"
img_col_names = ["Image0" ,"Image1", "Image2", "Image3"]

# initializing object that will process the csv's
set_builder = ProcessingTools.SetBuilder(img_col_names)
train_set, train_labels = set_builder.get_dataset(train_set_path)
validation_set, validation_labels = set_builder.get_dataset(validation_set_path)
test_set, test_labels = set_builder.get_dataset(test_set_path)

# initalizing the siamese network
siamese_net = SiameseNetwork().model
siamese_net.summary()
earlystopping = callbacks.EarlyStopping(monitor ="val_accuracy", 
                                        mode ="max", patience = 5, 
                                        restore_best_weights = True)

# setting optimizer and compiling model
siamese_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# training model
siamese_net.fit(
     [train_set[:,0], train_set[:,1]], train_labels[:], shuffle=True,
      validation_data=[[validation_set[:,0], validation_set[:,1]], validation_labels[:]], 
      batch_size=32, epochs=1, callbacks=[earlystopping]
      )

# evaluating
siamese_net.evaluate([test_set[:,0], test_set[:,1]], test_labels[:], batch_size=50)

# saving
siamese_net.save("Trained Simese Net")

# best run -- 86% Testing accuracy w dropout @ 0.2 and 3 batch normalization layers
