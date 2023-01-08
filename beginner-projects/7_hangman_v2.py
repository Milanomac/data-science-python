import random
from words import words
 
def get_word():
    word = random.choice(words)
    while '-' in word or ' ' in word:
        word = random.choice(words)
    
    return word.upper()

# creating a function to display the word with _ _ _
def play(word):
    word_completion = "_" * len(word) #initially it will display only underscores
    guessed = False
    guessed_letters = []
    guessed_words = []
    tries = 6

    print("Let's play hangman!")
    print(display_hangman(tries))
    print(word_completion)
    print('\n')
    while not guessed and tries > 0:
        guess = input("Please guess a letter or word: ").upper()
        if len(guess) == 1 and guess.isalpha():
            if guess is guessed_letters:
                print("You already guessed the letter", guess)
            elif guess not in word:
               print(guess, "is not in a letter in this word.")
               tries -= 1
               guessed_letters.append(guess)
            else:
               print("Good job,", guess, "is in the word!")
               guessed_letters.append(guess)

               # we update the variable word_completion reveal to the user all occurences of guess
               word_as_list = list(word_completion) # we do it by first, converting string to list  
               indices = [i for i, letter in enumerate(word) if letter == guess]

               # replacing each underscore at index with guess
               for index in indices:
                  word_as_list[index] = guess
               word_completion = "".join(word_as_list)
               if "_" not in word_completion:
                  guessed = True
        
        # this block is for checking if a word has already been guessed          
        elif len(guess) == len(word) and guess.isalpha():
            if guess in guessed_words:
               print('You already guessed that word', guess)
            elif guess != word:
               print(guess, "is not the word.")
               tries -= 1
               guessed_words.append(guess)
            else:
               guessed = True
               word_completion = word

        else:
            print("Not a valid guess.")
        print(display_hangman(tries))
        print(word_completion)
        print('\n')
    if guessed:
       print("Congrats, you guessed the word! You win!")
    else:
       print("Sorry, you ran out of tries. The word was " + word + ". Maybe next time!")

def display_hangman(tries):
    stages = [  # final state: head, torso, both arms, and both legs
                """
                   --------
                   |      |
                   |      O
                   |     \\|/
                   |      |
                   |     / \\
                   -
                """,
                # head, torso, both arms, and one leg
                """
                   --------
                   |      |
                   |      O
                   |     \\|/
                   |      |
                   |     / 
                   -
                """,
                # head, torso, and both arms
                """
                   --------
                   |      |
                   |      O
                   |     \\|/
                   |      |
                   |      
                   -
                """,
                # head, torso, and one arm
                """
                   --------
                   |      |
                   |      O
                   |     \\|
                   |      |
                   |     
                   -
                """,
                # head and torso
                """
                   --------
                   |      |
                   |      O
                   |      |
                   |      |
                   |     
                   -
                """,
                # head
                """
                   --------
                   |      |
                   |      O
                   |    
                   |      
                   |     
                   -
                """,
                # initial empty state
                """
                   --------
                   |      |
                   |      
                   |    
                   |      
                   |     
                   -
                """
    ]
    return stages[tries]

# main function that puts everything together

def main():
   word = get_word()      # we are getting the word using get_word function
   play(word)             # we pass this word to play function
   while input("Play again? (Y/N) ").upper() == "Y":
      word = get_word()      
      play(word)
   else:
      print("Thanks for playing!")             


main()