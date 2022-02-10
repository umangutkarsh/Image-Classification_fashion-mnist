# Image-Classification_fashion-mnist

Problem Statement: 
Imagine you’re a fashion designer, and your model has created 1000s of different shoes, shirts, accessories etc. Now I’ve decided to open an online website for my products, and so photos need to be taken for these products, and label them. Let’s say you have these products labelled but do you have these products labelled with the specifics the online platform demands? And if I suppose want to add more and more photos then it would be nice if the labelling can be automated and stream lined. Let’s say you have a website, where people can sell their clothes, but any person can join and upload what they want to sell, but these people don’t take their time to label their products. This can prove to be a problem for me, since the consumers may struggle to find the items, and the platform losses ground to competition. So, I decide to create a model that can assign these products their correct labels. This illustrates the significance of the image classification problem.

Preprocessing:
1.	Converting images to NumPy arrays.
2.	Neural networks don’t work with images files, they work with tensors, so all these have to be converted into numpy arrays.
3.	Preprocessing is simple since tensorflow objects are not used, and numpy arrays are being used.
4.	Image can be manipulated using a variable.


About IDX and CSV file format:
For any Beginner in the domain of Neural Network or Machine Learning, the most suitable data-set to get his/her hands dirty, is the MNIST Dataset.
But the first challenge that anyone would face before using the data in the images of the hand-written digits, in the data-set, is that the data-set is available in IDX format. Hence, the data needs to be converted to suitable format before we can use it in our code.
Now, let’s get to know what this file format

IDX file format is a binary file format.
Now, why store in this format when we have other text file formats?
The answer is performance and memory requirements.
If we look in terms of performance, binary file formats are far better than text file formats like CSV. CSV file formats are used to store tabular data, where for reading a particular value on a certain row or column, the software has to iterate over all the previous values. Whereas in binary file format you can literally store anything, provided you also write the proper information for parsing it, in the header of the file. It then makes accessing a binary file simpler and faster.

Also, storing data in binary format takes less memory, which is really an added advantage when a large volume of data needs to be stored.


Important Points:
1. An ideal kernel should have an odd number as the dimensions. So an essential quality of a kernel, is that it should be associated with a certain main pixel.
2. Working with numpy arrays have numerous benefits, like they can be easily rescaled.
3. In image processing, normalization is a process that changes the range of pixel intensity values. The purpose of dynamic range expansion in the various applications is usually to bring the image, or other type of signal, into a range that is more familiar or normal to the senses, hence the term normalization. 

4. train_images_array = train_images_array/255.0
   test_images_array = test_images_array/255.0

5. When working with numpy arrays, tensorflow can easily batch and shuffle and batch the dataset. This is done in the .fit() method. So, here the data pre-processing part gets over.

Dataset Info:

1. Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits. The original MNIST dataset contains a lot of handwritten digits. Members of the AI/ML/Data Science community love this dataset and use it as a benchmark to validate their algorithms. In fact, MNIST is often the first dataset researchers try. "If it doesn't work on MNIST, it won't work at all", they said. "Well, if it does work on MNIST, it may still fail on others." Zalando seeks to replace the original MNIST dataset
2. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.
3. To locate a pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27. The pixel is located on row i and column j of a 28 x 28 matrix. For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.
4. Labels 
Each training and test example is assigned to one of the following labels:
•	0 T-shirt/top
•	1 Trouser
•	2 Pullover
•	3 Dress
•	4 Coat
•	5 Sandal
•	6 Shirt
•	7 Sneaker
•	8 Bag
•	9 Ankle boot

TL;DR
•	Each row is a separate image
•	Column 1 is the class label.
•	Remaining columns are pixel numbers (784 total).
•	Each value is the darkness of the pixel (1 to 255)

Issues faced while Bulding the layers:
1.	Conv2D -> MaxPool2D -> Conv2D ->MaxPool ->Flatten(to unpack the tensor) ->Dense to output the layers representing the classes. (CNN architecture)
2.	In general, batch size may affect the speed of the training, but not the accuracy.
3.	When the BATCH_SIZE is parameter is set, tensorflow will automatically SHUFFLE and BATCH the numpy arrays
4.	L2 regularization can be done to prevent overfitting.
5.	First layer will be Conv2D not Dense, since that would unpack the images, into 1D vectors.
6.	Kernel size set to (5,5) since images, since the images are 28X28 -> 7x7 should be the max kernel size.
7.	Deciding the kernel no. is not always obvious what the correct no. should be
8.	Activation fn will be relu, since it behaves well in most situations.
9.	Conv2D is followed by a maxpool with kernel size 2,2 and stride = 2
10.	Since the first conv layer has 5, 5 dimensions, so the image will get scaled from 28x28 to 24x24.
11.	Since the maxpool layer only cuts the spacial dimensions, it will output a layer of 24x24x50 to 12x12x50.
12.	Having a kernel size of 3x3, reduces the number of parameters.
13.	 The output from the 4th layer is multidimensional, and it is to be made 1D and this can be done with Flatten layer, 
14.	Then Dense layer is put, which classifies into 10 numbers. A dense layer is a transformation, in which every output is a linear combination of the inputs. (Fully connected layer).
15.	With adam optimizer, accuracy on the test_set came out to be 90.27% , and on training_set it was 92.29% (also L2 regularization was applied)
16.	Not much improvement with SGD.
17.	After removing the regularizer, it performed much better.
18.	Not predicting correctly, some changes needed.
19.	Reducing max_pool to 1, removing the strides, filters early were set to 50, changed to 32,62, added some dropout layers and an additional dense layer after flattening
20.	No, when having two consecutive convolution layers can't be combined into one. The subsequent filter's inputs are the features extracted from the previous one. This results in the second layer's features are of higher-level than the previous.
21.	This is the basis of the whole CNN. Having multiple convolutional layers stacked along the depth of the network, allows the network to extract high-level features (not just edges and corners) from the input images.
22.	The first convolutional layer of a CNN is essentially a standard image filter (+ a ReLU). Its goal is to take a raw image and extract basic features from it (e.g. edges, corners). These are referred to as low-level features.
23.	The second convolutional layer, instead of the raw image, accepts the features extracted by the first as its input. This allows it to combine these basic shapes into more complex features.
24.	The features extracted become more and more complex as we go further down the network. Layers near the middle of the network extract the so called mid-level features, while the final layers extract high-level features.
25.	CNNs are powerful tools because it is trained to extract the best features for each task. This results in the network extracting different features for different tasks.
26.	Adding a kernel initializer, Usage of initializers Initializers define the way to set the initial random weights of Keras layers.
The keyword arguments used for passing initializers to layers depends on the layer. Usually, it is simply kernel_initializer and bias_initializer:
27.	And putting the filters as 32 and 64 respectively.
28.	While pre-processing the image, the np.expand_dims() is used to expand the dimension of dimension of images since the image is fed into batches, so this pre-processed image also needs to be converted into a batch, hence this method is applied. Also the batch dimension needs to be the first dimension, hence the axis=0
29.	Prediction not that much accurate, 92.250% maximum achieved, on the test_set
30.	Wasn’t giving accurate predictions on all samples.


CNN Intuition:

Steps for image processing: 
1.	Convolution – a mathematical operations on two functions which produces a third function.
2.	Max Pooling
3.	Flattening
4.	Full Connection

Convolution:
->Main idea is that feature detectors are used as filters for feature maps on the input image.
->The convolution function basically puts the image into a feature map while still preserving the spacial features. 
The step at which the feature detects are moved is called a stride.
Input Image + Feature detector = Feature Map (convolved feature).
As stride increase, size of output image decreases.
RELU(Step1-B):
The rectifier is applied to increase non-linearity, since images are themselves non-linear, and therefore the linearity should be broken

Max-Pooling:
Pooling is all about the neural network having a flexibility to recognize the image even if the image has distinctive features.
By pooling the image through a feature map, the features are still preserved, are also reducing 75% of the parameters preventing overfitting (since some information is irrelevant).
Image -> Convolution layer -> Pooling layer
Smears the noise. (Stride is the extent, with which the kernel moves down or right at each step.)

Flattening:
->Input layer is passed through a convolution function
->Then pooling is done
->Then the image is flattened, basically the pooled feature map is taken and converted into a column (taking the nums row by row and converting into cols) this flattened layer is passed through the artificial neural network.

Full Connection:
The ANN layer is added to the convolution network
The hidden layers here are called the fully connected layers, and in convolution network, the layer is fully connected hence it is called fully connected layers.
Combine the attributes into features.

->The final output layer listens to some particular neurons, upon which they predict the outcome. 
->Ex: The dog and the cat neuron learn on the weights of some particular neurons.

Summary: 
1.	A feature map is applied to the input image, called convolution.
2.	Then pooling is done, because of which the most of the features are get rid of, so it prevents overfitting.
3.	The pool is flattened convolution into a vector or column of all these values. (this is flattening), then the final connected layer is processed through network all this is passed through a process of forward propagation and back propagation(epochs).

# SoftMax and cross entropy functions:

1.	Softmax function is necessary to introduce to CNN. Softmax basically converts all the inputs taken into a probability distribution, so it is extensively used in classification tasks.
2.	The loss function should be minimized to maximize the performance of the model.
3.	Cross-entropy fn helps the NN to get to the optimal state. (better for classification)

Applying the transformation to the images of the training set, so that the images don’t overlearn or over-trained on the existing images.

Another reason why feature scaling is applied is that few algorithms like Neural network gradient descent converge much faster with feature scaling than without it. One more reason is saturation, like in the case of sigmoid activation in Neural Network, scaling would help not to saturate too fast.

Multiple Layers
Convolutional layers are not only applied to input data, e.g. raw pixel values, but they can also be applied to the output of other layers. The stacking of convolutional layers allows a hierarchical decomposition of the input.

# Building the CNN layers:

1.	Conv2D – No. of feature detectors(filters), kernel size(no. of feature detectors), activation function = As a consequence, the usage of ReLU helps to prevent the exponential growth in the computation required to operate the neural network. If the CNN scales in size, the computational cost of adding extra ReLUs increases linearly. , input shape(size of image).
2.	MaxPool2D – Applies pooling to avoid overfitting
3.	Flattening – flattening the layers to the ann model (a dense layer is a transformation, in which every output is the linear combination of inputs.)
4.	Full Connection – it is made to connect to the neural network
5.	Compile, fit (lower batch size improves accuracy)

# Computer Vision:
1.	Can have huge impact on many areas.
2.	Many tasks require awareness of surroundings
3.	Ability to understand what’s in the photo
->CNN basically a subtype of deep NN with one C-layer

KERNELS:
Basically acts as filters for image processing
CNN have the ability, to learn by themselves to detect feature maps.
1.	Convolution layer is simply a layer, in which we search for a pattern.
2.	Early stopping is a callback method which is called at the end of each epoch.
->The evaluated test accuracy, is always less than the validation accuracy
-> The test_accuracy is evaluated at the end of all work related to the model.
->validation accuracy is used to tune the different hyperparamters.
->essentially, we have tuned the hyperparamters to overfit on the validation set.
-> providing the softmax fn, directly into the layer is discouraged as it’s impossible to provide a numerically stable loss calculation for all models. To avoid this, incorporate the softmax into the loss function itself.

# Trainable and Non-trainable parameters:
1.	Trainable: The weights of our network, the parameters that the model is trying to learn.
#The data visualization tool for tensorflow is tensorboard.

Tensorboard is a callback method, and include it in the training process, it is to be included in the fit method, but the early_stopping should be the last element of the list, otherwise it bugs out.

To clean out tensorboard processes: 
taskkill /im tensorboard.exe /f
del /q %TMP%\.tensorboard-info\*

Hyperparamter tuning can also be done with tensorboard.

Training set: A set of examples used for learning, that is to fit the parameters [i.e., weights] of the classifier. Validation set: A set of examples used to tune the parameters [i.e., architecture, not weights] of a classifier, for example to choose the number of hidden units in a neural network.

Training data is the set of the data on which the actual training takes place. Validation split helps to improve the model performance by fine-tuning the model after each epoch. The test set informs us about the final accuracy of the model after completing the training phase.

Data logging is the process of collecting and storing data over a period of time in different systems or environments. It involves tracking a variety of events. Put simply, it is collecting data about a specific, measurable topic or topics, regardless of the method used.

# Confusion matrix with tensorboard:
x-axis: Predicted labels
y-axis: True labels
It is usually normalized (%) – Normalization is done for rows, not columns.

#Techniques for better performance of NN:
1.	Regularization: Overfitting jeopardizes NN. Early Stopping is sometimes not desirable since the process is still going on and the model is still on the process of obtaining valuable insights from the training set.
Regularization is a method used to reduce the capacity of the model to learn. It restricts the capacity of complex models, and encourages simpler representations. Technically, regularization means including some factors to the loss function to change its behaviour. Changing the loss affects the optimization of weights.
Penalize the behaviour: Loss can be increased when some unfavourable event occurs, so the model will adjust itself to not trigger that event as much.
Incentivize the behaviour: To decrease the loss, when we want the model to follow a behaviour.
2.	L2 regularization and weight decay: L2 reg is a type of regularization, L^2 = lambda($w^2). Purpose of L2 regularization is to simplify the model by scaling down all non-essential weights. In the case of SGD optimizer, weight decay and L2 reg, are indeed equivalent, but not generally.
Overfitting comes in different shapes and forms, L2 regularization is a method of decreasing the complexity of mode, and in case of SGD optimizer, L2 and weight decay are equivalent. 
3.	Dropout: Deliberately, dropping some of the neurons during the training process. This is done only during training, when evaluating the performance, all neurons are present.
Dropout is generally applied only to only fully connected layers. Since, the output are the linear combinations of the previous outputs, the dropouts basically scales the remaining outputs a bit to work properly.
4.	Data augmentation: A CNN can only learn patterns present on the training data.
Data augmentation is basically transforming the training data to artificially create more data. Through this, the purpose is not to increase the performance of the model, purpose is to expand the capabilities.

->Neural Network work with tensors.
-> So, images should be converted into numpy arrays, so that tensorflow can work with it.
->Images need to be resized, to involve less computational power.

Image.crop(), Image.filter(), Image.getbox(), Image.rotate().
Np.asarray().
Np.shape(image_array) -> (120, 90, 3) 3 represents RGB


