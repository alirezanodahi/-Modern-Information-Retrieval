import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from collections import defaultdict
#--------------------------------------------------------------------------------------
# Step 1: Preprocessing
def preprocess(document, stopwords_file):
    # Tokenization
    tokens = word_tokenize(document)

    # Removing punctuation and unwanted characters
    tokens = [word.lower() for word in tokens if word.isalnum()]

    # Normalization (converting to lowercase)
    tokens = [word.lower() for word in tokens]

    # Replacing integers and decimals with the word 'NUM'
    tokens = ['NUM' if word.isdigit() or word.replace('.', '', 1).isdigit() else word for word in tokens]

    # Removing stopwords
    with open(stopwords_file, 'r') as file:
        stop_words = set(file.read().splitlines())
    tokens = [word for word in tokens if word not in stop_words]

    # Root operation using Porter's algorithm
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]

    
    return " ".join(tokens)
#--------------------------------------------------------------------------------------
# Step 2: Extraction of Keywords
def extract_keywords(reference_documents):
    # Count word frequencies in all documents
    word_freq = {}
    for doc in reference_documents:
        for word in doc.split():
            word_freq[word] = word_freq.get(word, 0) + 1
            

    # Sort words by frequency in descending order
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    #print(word_freq)
    #print(sorted_words)
    # Select top 10% as keywords
    #print(len(sorted_words))
    keywords = [word for word, _ in sorted_words[:int(0.1 * len(sorted_words))]]
    print("saving keywords... ")
    #print(len(keywords))
    # Save the extracted keywords to a file
    with open('extracted_keywords(step2).txt', 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(keywords))
    print("saving keywords complete ")
    #print(len(keywords))
    return keywords
#--------------------------------------------------------------------------------------
# Step 3: Representation Space Vector
def create_vector_space_representation(reference_documents, inverted_index, keywords):
    # Create a matrix with rows as documents and columns as keywords
    #print(reference_documents)
    
    matrix = np.zeros((len(reference_documents), len(keywords)))
    z = len(reference_documents)
    
    # Fill in the matrix with TF-IDF values
    for i,doc in enumerate(reference_documents):
        #print('i = ',i)
        #print('doc = ' ,doc)
        for j,keyword in enumerate(keywords):
            #print('j = ',j)
            #print()
            #print('keyword = ' ,keyword)
            tf = doc.split().count(keyword)
            idf = np.log(len(reference_documents) / (1 + len(inverted_index[keyword])))
        
            tf_idf = tf * idf
            tf_idf = round(tf_idf , 4)
            if tf_idf == 0:
                tf_idf = int(tf_idf)
                
            matrix[i, j] = tf_idf
            # print('tf = ',tf)
            # print('idf = ' ,idf)
            # print("matrix[i, j] = tf * idf === ",matrix[i, j])
        
        print("create_vector_space_representation no : ",z )
        z = z - 1
    # Convert matrix to float64 data type(commen error!!!)
    matrix = matrix.astype(np.float64)
    
    #print(matrix.shape)
    # Save the matrix to a file
    np.savetxt('reference_documents_matrix(step3).txt', matrix)
    #print(matrix)
    return matrix
#up to hear every thing ok!!!
#--------------------------------------------------------------------------------------
# Modified Step 4: Document Retrieval
def retrieve_documents(test_document, reference_documents_matrix, inverted_index, keywords, k=5):
    # Create a matrix for the test document
    test_matrix = np.zeros((len(keywords)))
    #print(f"test_document = ",test_document)
    #print(f"reference_documents_matrix = ",reference_documents_matrix)
    #print(f"inverted_index = ",inverted_index)
    #print(f"keywords = ",keywords)
    #print(f"train_document_names = ",train_document_names)
    #print(f"train_documents = ",train_documents)
    #
    # Fill in the matrix with TF-IDF values for the test document
    for word in set(test_document.split()):
        #print(word)
        if word in keywords:
            print("find it in key word!! ")
            tf = test_document.split().count(word)
            #print("tf = " , tf)
            
            idf = np.log(len(reference_documents_matrix) / (1 + len(inverted_index[word])))
            print("idf = " ,idf)
            print("inverted_index[word] = ",inverted_index[word] ," ", len(inverted_index[word]))
            print()
            tf_idf = tf * idf
            tf_idf = round(tf_idf , 4)
            if tf_idf == 0:
                tf_idf = int(tf_idf)
            
            #print(test_matrix)
            #print("-"*30)
            #print(f"test_matrix[0, {keywords.index(word)}] = " ,tf_idf)
            test_matrix[keywords.index(word)] = tf_idf
            print(test_matrix[keywords.index(word)])
    # Convert test_matrix to float64 data type
    test_matrix = test_matrix.astype(np.float64)
    similarities = []
    
    # Calculate cosine similarity between the test document and reference documents.???T?
    for  i ,simi in enumerate(reference_documents_matrix):
        dot_product = np.dot(test_matrix,simi )
        print("simi = ",simi)
        print("test_matrix = ",test_matrix)
        #print("dot prudoct = ",dot_product)
        
        norm_vector1 = np.linalg.norm(test_matrix)
        norm_vector2 = np.linalg.norm(simi)
        # print("norm test  = ",norm_vector1)
        # print("norm simi = ",norm_vector2)
        cosine = dot_product / (norm_vector1 * norm_vector2 )
        cosine = round(cosine , 4)
        # print(cosine)
        similarities.append([i, cosine])
        #print("dot_product = ",dot_product)
        #print("norm_vector1 = ",norm_vector1)
        #print("norm_vector2 = ",norm_vector2)
        #print("cosine = ",cosine)
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    # print("similarities = ",sorted_similarities)

    # Get indices and names of the top k similar documents
    
    indices = sorted_similarities[:k]
    index = []
    #print("indices  sorted_similarities[:k][0] = " ,indices)
    for i in indices:
        idx = i[0]
        index.append(idx)
    #print(index)  
    #similar_document_names = [train_document_names[i] for i in index]
    
    #print("similar_document_names = " ,similar_document_names)
    #print("similar_documents = ",similar_documents)
    
    return indices
#--------------------------------------------------------------------------------------
# Create Inverted Index
def create_inverted_index(reference_documents):
    z = len(reference_documents)
    inverted_index = defaultdict(list)
    for doc_index, doc in enumerate(reference_documents):
        #print(doc_index)
        #print(doc)
        for term in set(doc.split()):
            inverted_index[term].append(doc_index)
            #print(inverted_index)
        print("left inverted_index : ",z)
        z = z -1
    return inverted_index
#--------------------------------------------------------------------------------------
# main:
print("loading data .... ")
train_folder = 'C:/Yelp Dataset/train'
test_folder = 'C:/Yelp Dataset/test'
stopwords_file = 'C:/Yelp Dataset/stopwords.txt' 
print("loading data complete")

print("Preprocessing .... ")
# Step 1: Preprocessing for the training set
train_files = [os.path.join(train_folder, file) for file in os.listdir(train_folder)]
z = len(train_files)

preprocessed_reference_documents = []
for file in train_files:
    with open(file, 'r', encoding='utf-8') as f:
        print(z)
        z = z - 1
        #print(f.read())
        preprocessed_reference_documents.append(preprocess(f.read(), stopwords_file))
        #print(preprocessed_reference_documents)
        
with open('preprocessed_document(step_1).txt', 'w', encoding='utf-8') as output_file:
    for word in preprocessed_reference_documents:
        output_file.write("".join(word))

print("Preprocessing complete")

print("Creating Inverted Index ...")
inverted_index = create_inverted_index(preprocessed_reference_documents)
#np.save("inverted_index(step2).npy", inverted_index)
print("Creating Inverted Index complete")

# Save the inverted index to a text file
print("saving inverted_index... ")
with open('inverted_index(step2).txt', 'w', encoding='utf-8') as output_file:
    for term, doc_indices in inverted_index.items():
        output_file.write(f"{term}: {', '.join(map(str, doc_indices))}\n")
print("saving inverted_index complete ")

print("Extraction of Keywords...")
# Step 2: Extraction of Keywords
keywords = extract_keywords(preprocessed_reference_documents)
print("Extraction of Keywords complete")

print("Representation Space Vector ...")
# Step 3: Representartion Space Vector
reference_documents_matrix = create_vector_space_representation(preprocessed_reference_documents, inverted_index, keywords)
print("Representation Space Vector complete")

# Step 4: Document Retrieval for the test set
print("Testing ... ")
test_files = [os.path.join(test_folder, file) for file in os.listdir(test_folder)]
train_file_names = [os.path.join(train_folder, file) for file in os.listdir(train_folder)]

#up to hare--------------------------------------------------------------------------------
# Create or open a file to save the results
output_file_path = 'C:/Yelp Dataset/output.txt' 
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for file_index, file in enumerate(test_files):
        #print("file_index = ",file_index)
        #print("file = ",file)
        
        with open(file, 'r', encoding='utf-8') as f:
            #preprocess in test doc....
            print(f"Preprocessing test document {file_index + 1 } ... ")
            preprocessed_test_document = preprocess(f.read(), stopwords_file)
            print("Preprocessing complete")
            #print(preprocessed_test_document)
            similar_documents_indices = retrieve_documents(preprocessed_test_document, reference_documents_matrix, inverted_index, keywords, k=5)
            index = []
            score = []
            #print("indices  sorted_similarities[:k][0] = " ,indices)
            for i in similar_documents_indices:
                idx = i[0]
                scr = i[1]
                index.append(idx)
                score.append(scr)
            
            #print(f"similar_documents_indices {file_index + 1 } = ",similar_documents_indices)
            #print(f"similar_document_names{file_index + 1 } = " , similar_document_names)
            #print(f"similar_documents{file_index + 1 } = " ,similar_documents)
            #print(f"retrieve_documents{file_index + 1 } = ",retrieve_documents)
            #print()
            # Display the similar documents
            output_file.write(f"Similar Documents for {file}:\n{'*'*80}\n")
            for i in range(len(index)):
                output_file.write(f"{i+1}. Train Document Name: {train_file_names[index[i]]} == score ===> {score[i]}\n{'-'*80}\n")
            output_file.write("\n")
                
            
            """for i, (doc_index, doc_name) in enumerate(similar_documents_indices):
                output_file.write(f"{i}. Train Document Index: {doc_index}, Train Document Name: {doc_name}\n{'-'*80}\n")
            output_file.write("\n")"""

print("Testing complete")
print(f"Results saved to {output_file_path}")

