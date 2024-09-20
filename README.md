# Modern-Information-Retrieval
this is a IR project based on yelp dataset that can use tf.idf and cosine similarity.
# Document Retrieval System

## Overview

This project implements a document retrieval system that processes a set of training documents, extracts keywords, and retrieves the most similar documents for a given test document. The system utilizes techniques such as tokenization, normalization, stemming, and TF-IDF for vector representation.

## Features

- **Preprocessing**: Tokenizes documents, removes punctuation and stopwords, and normalizes words.
- **Keyword Extraction**: Identifies the most frequently occurring words as keywords.
- **Vector Space Representation**: Constructs a matrix representing documents in a vector space using TF-IDF.
- **Document Retrieval**: Retrieves and ranks the most similar documents based on cosine similarity.

## Requirements

- Python 3.x
- NLTK library
- NumPy library
  
## Directory Structure

The project directory should be organized as follows:

- **`project-directory/`**: Root directory of the project
  - **`train/`**: Contains the training documents (place your text files here)
  - **`test/`**: Contains the test documents (place your text files here)
  - **`stopwords.txt`**: A text file containing stopwords to be removed during preprocessing
  - **`inverted_index(step2).txt`**: Output file that stores the inverted index
  - **`extracted_keywords(step2).txt`**: Output file that lists the extracted keywords
  - **`preprocessed_document(step_1).txt`**: Output file that contains the preprocessed training documents
  - **`output.txt`**: File that saves the results of the document retrieval process

This layout helps keep the project organized and ensures all necessary files are easily accessible.


## Usage

1. Place your training documents in the `train` folder.
2. Place your test documents in the `test` folder.
3. Provide a stopwords file named `stopwords.txt` in the root directory.
4. Run the script to preprocess documents, extract keywords, create an inverted index, and retrieve similar documents for each test document.

## Output

The results, including similar documents and their scores, will be saved in `output.txt`.

## Acknowledgments

This project uses the NLTK library for natural language processing and NumPy for numerical computations.

