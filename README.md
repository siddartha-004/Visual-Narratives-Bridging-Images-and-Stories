
# Visual Narratives : Bridging Images and Stories

This project is focused on generating descriptive stories based on image inputs. Using a combination of deep learning models for image captioning and text generation, this pipeline creates narratives (stories) from images. The project also includes analysis of the generated stories, such as sentiment analysis, word frequency, and sentence structure.


## Overview
This project takes an image as input and outputs a descriptive story based on that image. The image is processed to generate captions, which are then fed into a text generation model to create a coherent story. The stories can also be analyzed for sentiment, word frequency, and sentence length.
## Features

- Image Captioning: Extracts captions from images using trained models.
- Story Generation: Uses models like GPT-2 or Falcon to extend the captions into fully descriptive stories.
- Story Analysis: Performs sentiment analysis, word frequency, and sentence length analysis of the generated stories.


## Requirements
To run this project, you'll need the following libraries:

- transformers
- torch
- matplotlib
- seaborn
- pandas
- textblob
- nltk
- requests
- grammarbot


## Usage
Use MinorDemo.ipynb file for ouput with analysis.

This project can also be run directly in a Kaggle notebook. Follow these steps:

- Clone the Project: You can either upload the project files to your Kaggle environment.
- Install Dependencies: Kaggle comes with many libraries pre-installed, but to ensure all required packages are available, run the cells from top to bottom to ensure all packages are installed.
- Run the Code: To generate stories from images in Kaggle, ensure the image you want to process is uploaded to the Kaggle notebook environment.
- View the Output: After running the above code, the generated story and analysis (sentiment, word frequency, sentence length) will be displayed in the notebook, along with the plots for analysis.


## Architecture involved

![image](https://github.com/user-attachments/assets/5c476217-08df-499a-b7ee-9e3741f7da57)




## Roadmap

Setup and Dependencies:

    You start by setting the environment for Keras backend to TensorFlow.
    Load required libraries such as TensorFlow, Keras, and EfficientNet for image feature extraction.
    Install and unzip the Flickr8k dataset, which contains images and corresponding captions.

Dataset Loading:

    The load_captions_data function reads the caption text data and processes it to associate each image with its corresponding caption. Captions are standardized by adding start and end tokens ("<start>", "<end>").
    Images that don’t meet the minimum caption length requirements (5 words) are excluded from the dataset.

Dataset Splitting:

    The train_val_split function splits the dataset into training and validation sets.
    Captions are shuffled, and a specified percentage (default 80%) is used for training, with the remainder allocated to validation.

Text Preprocessing:

    TextVectorization is used to convert textual data (captions) into integer token sequences.
    The custom_standardization function performs text cleaning and tokenization, while removing unwanted characters from captions.

Image Preprocessing:

    Images are loaded, resized, and normalized via the decode_and_resize function.
    Image augmentation techniques like flipping, rotation, and contrast adjustment are applied to the training images.

Dataset Preparation:

    The make_dataset function creates TensorFlow datasets for both training and validation.
    Captions and corresponding images are mapped and processed into batches.

Model Architecture:

    CNN (EfficientNet): Used for extracting feature vectors from images.
    Transformer Encoder: The TransformerEncoderBlock processes these feature vectors with multi-head attention layers.
    Transformer Decoder: The TransformerDecoderBlock generates captions based on the extracted image features.

Training the Model:

    The ImageCaptioningModel class defines the custom training loop. It computes losses and gradients using backpropagation and updates model weights.
    Captions are generated using teacher forcing, where each word is predicted based on previous words and image features.
    A learning rate scheduler and early stopping criteria are implemented to enhance model training.

Caption Generation:

    Once the model is trained, the generate_caption function can be used to generate captions for new images by feeding them through the model.
    The generated captions are shown along with the corresponding images.
 Plotting

    Import necessary libraries (pandas, matplotlib).
    Create a DataFrame from training history and plot loss and val_loss.
    Plot acc and val_acc.

Story Generation Setup

    Define an API key and headers for Hugging Face API access.
    Create a function generate_story to generate stories using a specified model.
    Define theme-based prompts for different story genres.

Text Formatting and User Interaction

    Implement a function wrap_text for formatting text with specified line width.
    Create a function print_theme_options for user input on theme selection.

Model Loading and Story Generation

    Load a pre-trained GPT-2 model.
    Create functions to generate and preprocess stories, incorporating narrative hooks.

Grammar and Sentence Processing

    Implement functions for grammar checking, converting tenses, and removing angular brackets.
    Include functions for checking grammar using an external API.

Story Analysis

    Define an analyze_story function to perform sentiment analysis and extract word frequencies and sentence lengths.
    Create plotting functions for visualizing sentiment, word frequency, and sentence length distribution.

Confusion Matrix Calculation

    Initialize confusion matrix variables.
    Loop through true and predicted labels in chunks to accumulate the confusion matrix.
    Plot the confusion matrix using a heatmap.

Execution and Display

    Execute the main flow: loading models, generating stories, analyzing them, and plotting results.
    Display all generated plots.
## Results
![image](https://github.com/user-attachments/assets/1e5bf335-dfa3-442b-9a5b-afda71822a65)
![image](https://github.com/user-attachments/assets/d816fc97-ac6d-4986-b576-a1602a265f6f)
![image](https://github.com/user-attachments/assets/71f969a7-63cb-40ed-9ecb-c38e3cd696a7)


