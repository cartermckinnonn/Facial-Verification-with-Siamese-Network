# Facial-Verification-with-Siamese-Network
I used the LFW dataset to train a model with a Siamese Network architecture for the purpose of Facial Verification.

The overall goal of this project was to train a model using a Siamese Network architecture to be able to accurately predict whether two pictures of faces are of the same person. I decided to use a Siamese Network architecture because it generally performs better with smaller datasets. A Siamese network is a class of neural networks that contains one or more identical networks. We feed a pair of inputs to these networks. Each network computes the features of one input. And, then the similarity of features is computed using their difference or the dot product. In this project, I used two identical embedding layers and fed each of them one input image with the dimensions of (128x128). I then took the eucledian distance between those two outputs to get the final output. 

The process of finding the correct architecture for the embedding layer along with the dataset was a long one. I initially began this process by using the AT&T faces dataset, but I soon found that the dataset was much too small and was not suitable my project. I then found the Labeled Faces in the Wild (LFW) dataset and found the extra size much more suitable for my use case. Processing this dataset was another challenge I faced. All the images were held in a folder named lfw, and although the creators of the dataset included .txt files that contained the pairs for the train, validation, and testing sets, I still needed to transform the information in those txt files to something I could actually use. First, I transformed those txt files into csv files and read them into pandas. Then, I wrote a class of methods named "SetBuilder" that processed those csv's. The process of transforming the data continues as follows-
(the code can be found in the "ProcessingTools.py" file in the Siamese Network Folder) 
1. Read CSV into Pandas
2. Create an array of img pairs by using the included name and image number to create a file path to the image.
3. Use those filepaths to read the images from the lfw file, resize them to 128x128, and add a tuple of the image pair and the corresponding label to a numpy array. 
4. Randomize and return the dataset


For the embedding layer, I initially began by training my own deep Convolutional Neural Network from scratch, but I was not able to achieve satisfactory results from this. After doing more research, I discovered the process of transfer learning, and I implemented that process in my embedding layer. I used the pretrained Keras VGG16 model with "imagenet" weights. I froze all layers except for the last block, and I also added a new top for the model. The architecture can be seen in the "SiameseBuilder.py" file in the Siamese Network folder. I then used the Distance layer that I created to calculate the eucledian distance of the two outputs. 

After all of this, my model was able to acheive an 86% accuracy on the testing set. I used the Adam optimizer with the learning rate set at 0.00005, and I used a BinaryCrossentropy loss function. I also had 3 Batch Normalization layers and one dropout layer set at 0.2. The .jpg image in the repo is a screenshot of the final results from training.

To run the code, simply download the lfw file from the source below and put that file in the "Data" folder.

Sources
Dataset: http://vis-www.cs.umass.edu/lfw/

Research Paper: Heidari, Mohsen & Fouladi, Kazim. (2020). Using Siamese Networks with Transfer Learning for Face Recognition on Small-Samples Datasets. 1-4. 10.1109/MVIP49855.2020.9116915. 
