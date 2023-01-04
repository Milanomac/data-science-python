import random

print('\n--------------')
print('INSTRUCTIONS:\n--------------\nGame of rock paper scissors:\n')

while True: #this true loop is only to ask the user if he wants to play
    #again after the game (look at the end of the code)
 
    #we create a list with all possible choices

    choices = ['rock', 'paper', 'scissors']
    computer = random.choice(choices)

    #we initialize the player
    player = None

    #loop to prevent player from typing whatever he wants

    while player not in choices:
        player = input('Rock, paper, or scissors?: ').lower()

    if computer == player:
        print('Computer:',computer)
        print('Player:',player)
        print('It\'s a tie!')
    elif player == 'rock':
        if computer == 'paper':
            print('Computer:',computer)
            print('Player:',player)
            print('You loose!')
        if computer == 'scissors':
            print('Computer:',computer)
            print('Player:',player)
            print('You win!')
    elif player == 'scissors':
        if computer == 'paper':
            print('Computer:',computer)
            print('Player:',player)
            print('You win!')
        if computer == 'rock':
            print('Computer:',computer)
            print('Player:',player)
            print('You loose!')
    elif player == 'paper':
        if computer == 'rock':
            print('Computer:',computer)
            print('Player:',player)
            print('You win!')
        if computer == 'scissors':
            print('Computer:',computer)
            print('Player:',player)
            print('You loose!')

    play_again = input('Do you want to play again? (yes/no): ').lower()
    if play_again != 'yes':
        break
print('Bye!')