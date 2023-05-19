import random
print('\n--------------')
print('INSTRUCTIONS:\n--------------\nIn this game python will attempt to guess the number (integer) chosen by the user from the range from 1 to 100. PLease answer the following:\n')
def computer_guess(x):#już na początku user wie jaki jest jego numer, dlatego poniżej należy zdefiniować zakres przy użyciu zmiennych

    # definiujemy 3 zmienne i ich zakres
    low = 1
    high = x
    feedback = ''
    # jeśli komputer nie ma racji
    while feedback != 'c':
        # i jeśli dolny i górny zakres nie jest sobie równy 
        if low != high:
            #zgaduj dalej. We don't want the guess to be in the (1, x) range: We want to be able to change it according to the user's feedback
            guess = random.randint(low, high)
        feedback = input(f'Is {guess} correct, (C), too low, (L), or too high (H)?').lower()
        if feedback == 'h': #jeśli numer jest za duży, musimy dostosować zakres (zmniejszyć)
            high = guess - 1 #teraz zakres to (low, guess - 1)
        elif feedback == 'l':
            low = guess + 1

    print(f'The computer guessed your number {guess} correctly!')

computer_guess(100)





         


