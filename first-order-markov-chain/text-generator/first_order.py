import random
from collections import defaultdict

def build_markov_chain(text):
    words = text.split()
    markov_chain = defaultdict(list)

    for i in range(len(words) - 1):
        current_word = words[i]
        next_word = words[i + 1]
        markov_chain[current_word].append(next_word)

    #probabilities in the dictionary
    for current_word, next_words in markov_chain.items():
        probabilities = [next_words.count(word) for word in next_words]
        total_prob = sum(probabilities)
        probabilities = [prob / total_prob for prob in probabilities]
        markov_chain[current_word] = (next_words, probabilities)

    return markov_chain

def generate_sentence(markov_chain, length=10):
    current_word = random.choice(list(markov_chain.keys()))
    sentence = [current_word]

    for _ in range(length - 1):
        next_words, probabilities = markov_chain.get(current_word, ([], []))
        if not next_words:
            break  # Break if no candidates
        next_word = random.choices(next_words, weights=probabilities)[0]
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)

file = 'hamlet.txt' if input("Enter 1 for hamlet Enter 2 for sa'adi\n") == '1' else 'saadi_norm.txt'

with open(file, 'r', encoding='utf-8') as file:
    text_data = file.read()

markov_chain_model = build_markov_chain(text_data)

# Generate
for _ in range(20):
    generated_sentence = generate_sentence(markov_chain_model, length=10)
    print(generated_sentence)
