import random
from words import words
import string # predetermined list of english letters from alphabet
 
def get_word(words):
    word = random.choice(words)
    while '-' in word or ' ' in word:
        word = random.choice(words)
    
    return word.upper()

def hangman():
    word = get_word(words)                         
    word_letters = set(word)                              
    alphabet = set(string.ascii_uppercase)                  
    used_letters = set()
    tries = 10                                   

    while len(word_letters) > 0:
        
        # letters used
        # .join(['a', 'b', 'cd']) --> 'a b cd'
        print('You have guessed these letters: ', ' '.join(used_letters))

        word_list = [letter if letter in used_letters else '_' for letter in word]
        print('Current word: ', ' '.join(word_list))

# user input

        user_letter = input('Guess a letter: ').upper()
        if user_letter in alphabet - used_letters:              # we look for element within a combined set
            used_letters.add(user_letter)                       # and if it is there we add it to used letters
            if user_letter in word_letters:                     # and if user types the correct letter (the one that is in our word)
                word_letters.remove(user_letter)                # then we can remove it from the set

        elif user_letter in used_letters:
            print('You have alreay used that caracter. Please try again!')

        else:
            print('Invalid character. Please try again')

hangman()