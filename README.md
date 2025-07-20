# Image_classification

COMPANY : CODTECH IT SOLUTIONS

NAME : SHREYA YADAV

INTERN ID : CT04DG3452

DOMAIN : MACHINE LEARNING

DURATION : 4 WEEKS

MENTOR : NEELA SANTOSH

🖼️ Image Classification Using CNN on CIFAR-10

This project demonstrates a practical implementation of an image classification model using Convolutional Neural Networks (CNNs) on the CIFAR-10 dataset, a well-known benchmark in computer vision. The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 different categories such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class has 6,000 images, with 50,000 for training and 10,000 for testing.

📦 Dataset Preprocessing
The dataset is loaded using tensorflow.keras.datasets.cifar10. The first step involves normalizing the pixel values of the images by scaling them to a range between 0 and 1. This improves the convergence speed of the neural network during training. Additionally, the labels are one-hot encoded using to_categorical() to prepare them for multiclass classification.

🧠 Model Architecture
The model is built using Keras Sequential API and consists of the following layers:

Convolutional Layers: The network begins with three convolutional layers:
First layer with 32 filters, 3x3 kernel, and ReLU activation.
Followed by two more convolutional layers with 64 filters each.
Pooling Layers: After the first and second convolutional layers, MaxPooling2D layers are added to reduce spatial dimensions and extract dominant features.
Flattening Layer: Converts the 3D feature maps into a 1D vector before feeding into dense layers.
Fully Connected (Dense) Layers:
One hidden dense layer with 64 neurons and ReLU activation.
Output layer with 10 neurons (one for each class) and softmax activation to generate probability distributions.
This architecture is effective for capturing spatial hierarchies in image data while being lightweight enough for fast training on standard hardware.

⚙️ Model Compilation and Training
The model is compiled using the Adam optimizer, with categorical crossentropy as the loss function and accuracy as the evaluation metric. The training is conducted over 10 epochs, allowing the model to learn patterns in the image data and improve its classification performance.

The fit() method trains the model on the training data and tracks its performance on the validation set. The training history, including accuracy and loss values per epoch, can be visualized using Matplotlib to analyze convergence and performance trends.

📈 Evaluation and Results
After training, the model is evaluated on the test dataset. It achieves a solid performance given the simplicity of the architecture, typically reaching 70–80% accuracy, depending on hardware and number of epochs.

This performance showcases how CNNs can effectively learn from image data even without complex architectures like ResNet or VGG. The project also highlights the value of using standardized datasets like CIFAR-10 for benchmarking and experimentation.

💡 Key Takeaways
CNNs are highly effective for image classification tasks due to their ability to capture local features through convolution.
Proper preprocessing (normalization and one-hot encoding) is critical for training stable and accurate models.
Even with a relatively shallow network, high accuracy can be achieved on well-structured datasets like CIFAR-10.
