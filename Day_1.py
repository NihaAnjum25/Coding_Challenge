def tokenize_text(paragraph):  

    paragraph = paragraph.lower()
    cleaned_text = ""
    for char in paragraph:
        if char.isalnum() or char.isspace():
            cleaned_text += char
        else:
            cleaned_text += " "
    
    tokens = cleaned_text.split()
    
    return tokens


def count_word_frequency(words):

    frequency = {}
    
    for word in words:
        if word in frequency:
            frequency[word] += 1
        else:
            frequency[word] = 1
    
    return frequency


def remove_stopwords(words, stopwords):

    filtered_words = []
    
    for word in words:
        if word not in stopwords:
            filtered_words.append(word)
    
    return filtered_words

if __name__ == "__main__":
    
    sample_paragraph = """
    This is the first coding challenge of the 60-day coding challenge from ABTalksOnAI. 
    The goal is to practice basic text processing techniques such as tokenization, 
    counting word frequency, and removing stopwords. 
    """
    
    # Define a basic stopword set
    stopword_set = {
        "is", "the", "and", "of", "in", "to", "a", "an", "are"
    }
    
    # Step 1: Tokenize
    tokens = tokenize_text(sample_paragraph)
    print("Tokens:")
    print(tokens)
    
    # Step 2: Remove Stopwords
    filtered_tokens = remove_stopwords(tokens, stopword_set)
    print("\nFiltered Tokens (Stopwords Removed):")
    print(filtered_tokens)
    
    # Step 3: Word Frequency
    word_freq = count_word_frequency(filtered_tokens)
    print("\nWord Frequency:")
    print(word_freq)