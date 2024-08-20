#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[1]:


pip install torch transformers spacy


# In[2]:


pip install nltk


# In[3]:


import nltk
nltk.download('punkt')


# # Pre-processing step using Tokenizer

# In[4]:


from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

text = "This is a sample text. It contains multiple sentences. We want to tokenize it."

# Tokenize into sentences
sentences = sent_tokenize(text)

# Tokenize each sentence into words
tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

print(sentences)
print(tokenized_sentences)


# # Pre-processing step using *Stemming*

# In[5]:


from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Sample text
text = "This is a sample text. It contains multiple sentences. We want to tokenize it."

# Tokenize the text into words
words = word_tokenize(text)

# Initialize the Porter Stemmer
porter_stemmer = PorterStemmer()

# Apply stemming to each word
stemmed_words = [porter_stemmer.stem(word) for word in words]

print(stemmed_words)


# # pre processing step convert it to Token Id

# In[6]:


from transformers import AutoTokenizer

# Initialize the tokenizer with the model you are using

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

def preprocess_input(text, max_length=128):

    # Add the "paraphrase: " prefix to the input text

    formatted_text = f'paraphrase: {text}'

    # Tokenize the input text

    input_ids = tokenizer(formatted_text, return_tensors="pt", padding="longest",
                          max_length=max_length, truncation=True).input_ids

    return input_ids

# Example input text

input_text = "Please paraphrase this sentence."

# Preprocess the input text

input_ids = preprocess_input(input_text)

# Print the pre-processed input IDs

print(input_ids)


# In[7]:


# Example input text
input_text = "Generating text is the task of producing new text. These models can, for example, fill in incomplete text or paraphrase."

# Preprocess the input text
input_ids = preprocess_input(input_text)

# Print the pre-processed input IDs
print(input_ids)


# # Trained the Model

# In[8]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to(device)


# # SVO Arrangement

# In[9]:


def organize_output(paraphrase_output):
    # Logic to organize paraphrase output into a structured sentence
    # Example logic: Split the output into tokens and organize them into SVO structure
    tokens = paraphrase_output.split()

    subject = " ".join(tokens[:3])
    verb = " ".join(tokens[3:6])
    obj = " ".join(tokens[6:])

    structured_sentence = f"{subject} {verb} {obj}"

    return structured_sentence


# ## Define the Paraphrase function

# In[10]:


def paraphrase(
    question,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=128
):
    input_ids = tokenizer(
        f'paraphrase: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids

    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )

    paraphrases = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Organize each paraphrase into a structured sentence
    structured_sentences = [organize_output(paraphrase) for paraphrase in paraphrases]

    return structured_sentences


# # Examples for paraphrasing text or input

# In[11]:


text = 'Shivam is doing Data science.'

structured_sentences = paraphrase(text)
print(structured_sentences)


# In[12]:


text = 'Ram eat mango'

structured_sentences = paraphrase(text)
print(structured_sentences)


# In[13]:


text = 'I have done this work '

structured_sentences = paraphrase(text)
print(structured_sentences)


# In[ ]:




