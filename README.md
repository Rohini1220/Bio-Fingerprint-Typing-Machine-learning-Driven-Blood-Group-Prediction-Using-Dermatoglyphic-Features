![image](https://github.com/user-attachments/assets/98ee8ef3-6613-4479-b042-0b5054d2a450)
![image](https://github.com/user-attachments/assets/fd2c86aa-09ca-4f04-a0ac-864e57a90c82)
![image](https://github.com/user-attachments/assets/f956d71f-c2f9-4362-aad6-be116187441e)
![image](https://github.com/user-attachments/assets/3f71d66a-8468-4536-9e3f-3b5452e0afb8)
Abstract— This research employs a CNN model to classify blood types from fingerprint photos. Fingerprint pictures from eight blood groups A+, A-, B+, B-, AB+, AB-, O+, and O- are included in the datasets. To improve image quality, the preprocessing step uses Canny edge detection, Gaussian blurring, and histogram equalisation. To ensure better model generalisation, augmentation and normalisation are done using an Image Data Generator. Multiple convolutional and pooling layers make up the CNN model, which is used to extract pertinent characteristics from fingerprint images. Automated blood group classification using fingerprint photos is made possible by saving the trained model for use in future predictions. By offering a non-invasive technique for blood type identification, this strategy improves healthcare accessibility and advances biometric-based medical applications.
Keywords—Blood Group Classification, Non-Invasive Blood Group Detection, Data Augmentation, TensorFlow & Keras, Fingerprint Recognition, CNN, Image Processing, Histogram Equalisation, Gaussian Blur, Canny Edge Detection, Deep Learning, Biometric Identification, Medical Image Analysis

I.Introduction
In forensic applications, transfusion safety, and medical diagnostics, blood group classification is essential. Conventional blood typing techniques include intrusive procedures and chemical reagents, which can take a long time and call for skilled personnel. A different method for classifying blood groups using biometric data like fingerprints is provided by recent developments in deep learning and computer vision. The goal of this research is to create a deep learning-based, non-invasive system that uses CNN to accurately identify blood types from fingerprint scans.

To improve fingerprint photos for improved feature extraction, image processing techniques like Canny edge detection, Gaussian blur, and histogram equalisation are used. A dataset of fingerprint photos with various blood group labels is used to train a CNN-based deep learning model. To enhance generalisation and make sure the model works well on unseen samples, data augmentation and normalisation are used.

This method offers a quick and automated way to identify blood groups while lowering reliance on conventional blood
testing, which has enormous promise in the healthcare industry. The system can offer a dependable, non-invasive, and scalable solution for medical applications by combining deep learning and image processing. Larger datasets, better model architectures.

III.PROPOSED METHODOLOGY

1.Gathering and preprocessing datasets
    
 Gathering a dataset of fingerprint photos with matching blood group labels is the first stage in our suggested process. Preprocessing methods are used to improve the quality of raw photographs because they may contain noise and illumination changes. To enhance contrast, lower noise, and draw attention to key features, techniques including Canny edge detection, Gaussian blur, and histogram equalisation are employed. To guarantee consistency in model input, all photos are also normalised and scaled to a set dimension of 128x128 pixels.

2.Data Splitting and Augmentation 
   Rotation, flipping, scaling, and other data augmentation techniques are used to increase the model's generalisation and avoid overfitting. This guarantees that the model picks up a variety of patterns from the fingerprint pictures.
3.DL Model
   A CNN is made to classify blood groups based on fingerprints. The model uses max-pooling layers to minimise spatial dimensions after several convolutional layers for feature extraction. The final output layer classifies images into one of the eight blood types using the softmax activation function, while hidden layers employ the ReLU activation function to add non-linearity. To increase learning efficiency, the model is optimised using the Adam optimiser after being trained using the categorical cross-entropy loss function.
4.Training and Assessing Models
To maximise performance, the enhanced dataset with several epochs is used to train the CNN model. To determine how successfully the model generalises to new data, accuracy and loss measures are calculated as part of the evaluation process. Techniques like dropout layers are employed to avoid overfitting, while hyperparameter adjustment is done to maximise model performance. The model is saved and ready for deployment as soon as it produces findings that are satisfactory.
5. Implementation Utilizing Streamlit
Streamlit, an interactive web platform, is used to deploy the trained CNN model for real-time blood group prediction. Users can upload fingerprint photos through an easy-to-use interface, and the backend system processes them. The relevant blood group is then predicted by the trained CNN model after analysing the fingerprint data. Users may quickly and accurately determine their blood group thanks to the Streamlit interface, which guarantees a seamless, real-time experience with easy accessibility.
A.Model Architechure
Multiple layers make up the proposed CNN, which is intended to extract pertinent fingerprint information and categorise them into blood groups. A MaxPooling2D layer that lowers the spatial dimensions while maintaining significant features comes after the Conv2D layer, which starts the model with 32 filters of size (3x3). In order to gradually acquire increasingly complicated fingerprint patterns, this pattern repeats with increasing filter sizes.The ReLU activation function is applied by each convolutional layer, guaranteeing non-linearity and effective feature learning. After each convolution, the spatial dimensions are cut in half by the max-pooling layers (2x2).
Following feature extraction, the model employs a Flatten layer to transform a 25,088-size 1D vector. A fully connected Dense layer with 128 neurones receives this vector and uses it to learn high-level representations. The fingerprint is categorised into one of the eight blood groups by a final dense layer comprising eight neurones that uses the softmax activation function. To ensure effective learning, the model's 3,305,672 trainable parameters are optimised using Adam optimiser.This deep architecture is appropriate for fingerprint-based blood group prediction since it improves feature detection and classification accuracy.
B.Training Process
A collection of fingerprint pictures labelled with relevant blood types was fed into the proposed CNN model during the training phase. Since there were several classes in the classification challenge.Faster convergence was ensured by the effective weight updates provided by the Adam optimiser. Data augmentation methods like rotation, flipping, and zoom were used to improve generalisation and avoid overfitting after the dataset was divided into training and validation sets. 

Batch normalisation was used to stabilise learning during the model's multi-epoch training. In order to achieve the best possible balance between underfitting and overfitting, performance indicators like as accuracy and loss were tracked throughout training. Early stopping was then used to terminate training when validation loss stopped getting better.
IV. Workflow
1. Preparing and augmenting data
Gathering and preparing fingerprint photos is the initial stage in the project procedure. Preprocessing guarantees consistency in the dataset because raw photos can differ in size, orientation, and quality. For consistency across the dataset, the photos are scaled to a fixed size of 128x128 pixels. To further enhance the model's convergence during training, pixel values are normalised. Rotation, flipping, zooming, and brightness modifications are examples of data augmentation strategies used to improve the model's generalisation ability and avoid overfitting. By increasing the dataset's diversity without requiring more data gathering, this step enables the model to pick up more reliable features.
 
                             
2. Training and acessing models

A CNN uses the preprocessed dataset for training. Three convolutional layers make up the CNN architecture, and each is succeeded by max pooling layers that extract pertinent spatial data. Intricate information in fingerprint photos that correspond to blood group classification are captured by these layers. To create the final classification, the retrieved features are flattened and then run through dense layers that are fully connected. The output layer predicts the probability distribution of various blood types using the softmax activation function. 
Gathering and preparing fingerprint photos is the initial stage in the project procedure. Preprocessing guarantees consistency in the dataset because raw photos can differ in size, orientation.
3. Implementation Utilising Streamlit

Streamlit, a lightweight and interactive web framework, is used to deploy the model once its performance is satisfactory. Users can contribute fingerprint pictures for real-time blood group classification using the trained model incorporated into an intuitive web interface. The preprocessing pipeline is used when a picture is submitted, and then model inference is used to produce a prediction. The outcome is shown on the Streamlit interface quickly, giving users a smooth and simple experience. Additionally, the interface gives users the ability to upload multiple photographs, examine categorisation confidence scores, and effectively engage with the system. This deployment strategy makes the system usable for practical real-world applications without necessitating sophisticated technical knowledge.
V. DATA ACQUISTION

1. Selection of data sources

The calibre and variety of  data set training are critical to achieve  any deep learning model. In this project, fingerprint images were selected as the primary input for blood group classification. High-resolution fingerprint scans from medical datasets and publicly accessible biometric databases served as the dataset's sources. Furthermore, synthetic data generation techniques were taken into consideration to augment real-world data in situations where a suitable dataset was not easily accessible. This made sure there was enough variation in the dataset to increase the model's generalisation and resilience.

2. Gathering and Curating Images

A curation method was applied to the gathered fingerprint photos in order to eliminate any samples that were poor quality, hazy, or lacking. Only high-resolution photos were kept for additional processing because fingerprint patterns are extremely sensitive to image clarity. To build a well-structured dataset, each image was meticulously labelled with the matching blood group. A diverse and representative dataset was ensured by collecting photos from people of various ages, skin tones, and ethnic origins in order to prevent any potential biases.

3. Normalisation and Preprocessing

Uneven orientations, noise, and changes in lighting are common in raw fingerprint photographs. Preprocessing methods were used to overcome these obstacles. In order to preserve consistency and lower computational cost, each image was scaled to 128x128 pixels. To make fingerprint ridges more visible, contrast enhancement methods such histogram equalisation were applied. To make sure the model learns effectively and is not biassed towards different intensity levels, pixel values were also normalised between 0 and 1.

4. Agumenting data to improve learning

Data augmentation methods were used to increase the model's capacity to generalise across various fingerprint variants. Random rotation, flipping, zooming, and brightness modifications were among the augmentation techniques used to fictitiously expand the dataset and add diversity. By making sure the model learnt to identify key characteristics in various fingerprint images rather than memorising particular patterns, this step helped avoid overfitting. In order to make the model resilient and able to deal with real-world situations where fingerprint pictures might not always be correctly collected, augmentation was essential.

4.Dividing the Training and Testing Datasets

To assess the model's ultimate correctness and make sure it could effectively generalise to new data, the test set was kept apart. The model was successfully trained and evaluated on a separate dataset by using a standard split of 70% training, 20% validation, and 10% testing.

SYSTEM ARCHITECTURE:

The Blood Group Classification System's system architecture uses a CNN to forecast blood types by processing fingerprint pictures in an efficient manner according to a controlled workflow. The CNN Model, Image Preprocessing Module, Feature Extraction Module, Trained Model Storage, and Streamlit UI are some of the main parts of the system. A user submits a fingerprint image via the Streamlit UI to start the procedure. In order to increase classification accuracy, the Image Preprocessing Module subsequently improves the image quality by shrinking, normalising, and eliminating noise. The image is transmitted to the Feature Extraction Module following preprocessing, where it finds and extracts crucial features including ridges, patterns, and minute details. The CNN model uses these features as input to categorise the fingerprint into a matching blood group.


A sizable collection of fingerprint photos with blood group labels is used to train the CNN model. Prior to going through several convolutional, pooling, and fully connected layers in the CNN architecture, the dataset is preprocessed to standardise the pictures during training. The trained model can be reloaded for real-time classification because it is saved as blood_group_model.h5. Following processing of the fingerprint image, the system outputs the anticipated blood group, which the user can view using the Streamlit user interface. Scalability is guaranteed by this modular architecture, which also permits future enhancements like adding more biometric data or improving feature extraction methods for increased precision.

VI  RESULTS AND DISCUSSION

With a training accuracy of 84.28% and a loss of 0.5274, the suggested deep learning model for blood group categorisation was successful. With a validation accuracy of 84.88% and a validation loss of 0.4802, the model continued to perform competitively throughout validation. These findings show that the model successfully learns patterns from the training dataset and generalises well to new data. The model's dependability in practical applications is demonstrated by the consistent decline in loss throughout epochs, which indicates that it converged well without experiencing appreciable overfitting.
Based on the input features, the model successfully differentiates between various blood groups, as shown by the classification performance. Minor variations in accuracy, however, raise the possibility that some blood group types were incorrectly categorised because of feature representation similarities. These small errors could have been caused by a number of things, including differences in sample quality, an uneven distribution of data, or overlapping characteristics between various blood types. Model performance could be further improved by addressing these constraints by using augmentation techniques and more balanced datasets.

Metric	Value
Training Accuracy	0.8428
Training Loss	0.5274
Validation Loss	0.4802


All things considered, the model's high accuracy and minimal validation loss attest to its efficacy in blood group classification. The findings imply that automated blood group detection, which is essential for transfusion procedures and medical diagnostics, may benefit from deep learning-based classification. To further increase classification accuracy and lower misclassification errors, future developments might involve growing the dataset, adjusting hyperparameters, and incorporating more deep learning methods.

VII  CONCLUSION

The promise of automated categorisation using convolutional neural networks was effectively illustrated by the deep learning-based blood group classification research. The model demonstrated outstanding generalisation skills, making it appropriate for practical applications, with training and validation accuracy of 84.28% and 84.88%, respectively. The outcomes demonstrate how well deep learning can identify blood types, which is essential for transfusion safety and medical diagnostics. Even though the model worked effectively, a few minor misclassifications show that more dataset balancing and augmentation methods are required to increase accuracy. For even greater performance, future research can concentrate on adjusting hyperparameters, adding more data sources, and utilising cutting-edge deep learning architectures. This work opens the door for more effective and precise blood group classification systems by reaffirming the importance of AI-driven healthcare solutions.
