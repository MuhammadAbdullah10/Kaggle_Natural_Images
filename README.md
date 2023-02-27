  ---
# Dataset

The dataset used in the code is called "[Natural Images](https://www.kaggle.com/datasets/prasunroy/natural-images)" and can be found on **Kaggle**. It is a collection of eight categories of images that are found in the natural world, including:

  1.	**Airplane**
  2.	**Car**
  3.	**Cat**
  4.	**Dog**
  5.	**Flower**
  6.	**Fruit**
  7.	**Motorbike**
  8.	**Person**

The images in the dataset are not of a fixed size or resolution, and vary in size and shape. There are a total of **7,251** images in the dataset, with roughly **700** images per category.
 
 ---
# Code

The code is written in **Python** and uses various libraries for loading, processing, and training the dataset. Here is a detailed explanation of the code:
1.	The required libraries are imported at the beginning of the code:

	* **TensorFlow**: a popular open-source machine learning framework

	* **NumPy**: a library for working with arrays and matrices

	* **Matplotlib**: a plotting library for creating visualizations
  
	* **split_folder**: a library for splitting a folder into train, validation, and test sets
  
	* **ImageDataGenerator**: a class for generating batches of image data with real-time data augmentation
  
2.	The location of the input folder is specified, which is the folder where the Natural Images dataset is stored.
3.	The location of the output folder is specified, which is the folder where the **train**, **validation**, and **test** sets will be saved.
4.	The "**ratio**" function from the "**split_folder**" library is used to split the dataset into **training**, **validation**, and **test** sets with a ratio of **70%**, **15%**, and **15%**, respectively.
5.	The size of the images is specified to be **150 x 150** pixels.
6.	The paths to the training, validation, and test sets are specified using the output folder from step 3.
7.	An **ImageDataGenerator** is created for the training set, which **rescales** the pixel values to be between **0** and **1**, applies random shearing, zooming, and horizontal flipping to the images for data augmentation.
8.	An **ImageDataGenerator** is created for the test and validation sets, which only rescales the pixel values to be between **0** and **1**.
9.	A training set and test set are generated from the ImageDataGenerators, with a **batch size** of **32** and "**categorical**" class mode, which means that the labels are one-hot encoded.
10.	A **Sequential model** is created using the **Keras** API from **TensorFlow**.
11.	Four **convolutional layers** are added to the model with **32**, **64**, **128**, and **128** filters, respectively, with a kernel size of **3 x 3** and "**relu**" activation function.
12.	Four **MaxPool2D** layers are added to the model, each with a pool size of **2 x 2**.
13.	A **Flatten layer** is added to the model to flatten the output from the convolutional layers.
14.	A **Dense layer** is added to the model with **512** units and "**relu**" activation function.
15.	An **output layer** is added to the model with **8** neuron and "**sigmoid**" activation function, which represents the 8 categories in the [**Natural Images dataset**](https://www.kaggle.com/datasets/prasunroy/natural-images).
16.	The model is compiled with "**categorical_crossentropy**" loss function, "**adam**" optimizer, and "**accuracy**" metric.
17.	The model is **trained** using the **fit_generator** method from **TensorFlow**, with the **training_set**, **validation_data**, **epochs**, **steps_per_epoch**, and **validation_steps** as arguments. The history of loss and accuracy is saved in **r**.
18.	The **loss** and **accuracy** of the model are plotted using **Matplotlib** to visualize the performance of the model during **training** and **validation**.
19.	The **test_loss** and **test_acc** are evaluated using the evaluate method from **TensorFlow** on the **test_data**.[^1]
[^1]:Note: The accuracy of the model can be improved by tuning the hyperparameters, adding more layers, or using a different model architecture.

