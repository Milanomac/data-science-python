import tkinter
from tkinter import *
import customtkinter

import random

def next_turn(row,column):
    
    global player # accesing player

    if buttons[row][column]['text'] == "" and check_winner() is False:

        if player == players[0]:

            buttons[row][column]['text'] = player

            if check_winner() is False:
                player = players[1]
                label.config(text=(players[1]+" turn"))
            
            elif check_winner() is True:
                label.config(text=(players[0]+' wins'))

            elif check_winner() == "Tie":
                label.config(text=("Tie"))

        else:

            buttons[row][column]['text'] = player

            if check_winner() is False:
                player = players[0]
                label.config(text=(players[0]+" turn"))
            
            elif check_winner() is True:
                label.config(text=(players[1]+' wins'))

            elif check_winner() == "Tie":
                label.config(text=("Tie!"))

# defining all the win conditions, return true if somebody won, false if lost, and tie if its a tie
def check_winner():

# checking vertical conditions 
    for row in range(3):
        if buttons[row][0]['text'] == buttons[row][1]['text'] == buttons[row][2]['text'] != "":
            buttons[row][0].config(bg='darkblue')
            buttons[row][1].config(bg='darkblue')
            buttons[row][2].config(bg='darkblue')
            return True
    
    for column in range(3):
        if buttons[0][column]['text'] == buttons[1][column]['text'] == buttons[2][column]['text'] != "":
            buttons[0][column].config(bg='darkblue')
            buttons[1][column].config(bg='darkblue')
            buttons[2][column].config(bg='darkblue')
            return True

# checking diagonal conditions
    if buttons[0][0]['text'] == buttons[1][1]['text'] == buttons[2][2]['text'] != "":
        buttons[0][0].config(bg='darkblue')
        buttons[1][1].config(bg='darkblue')
        buttons[2][2].config(bg='darkblue')
        return True

    elif buttons[0][2]['text'] == buttons[1][1]['text'] == buttons[2][0]['text'] != "":
        buttons[0][2].config(bg='darkblue')
        buttons[1][1].config(bg='darkblue')
        buttons[2][0].config(bg='darkblue')
        return True

    elif empty_spaces() is False:
        for row in range(3):
            for column in range(3):
                buttons[row][column].config(bg='violet')

        return "Tie"
    
    else:
        return False

def empty_spaces():
    
    spaces = 9

    for row in range(3):
        for column in range(3):
            if buttons[row][column]['text'] != "":
                spaces -= 1

    if spaces == 0:
        return False
    
    else:
        return True

def new_game():

    global player

    player = random.choice(players)

    for row in range(3):
        for column in range(3):
            buttons[row][column].config(text="",bg="darkgrey")


# ===========================
# Creating GUI for the game
# ===========================
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "darkblue", "dark-blue"

window = customtkinter.CTk()
window.geometry("800x700")
window.title("Tic-Tac-Toe")
players = ["x","o"]
player = random.choice(players)
buttons = [[0,0,0],
           [0,0,0],
           [0,0,0]]
label = Label(text= player + " turn", font=('consolas',40), bg='grey')

label.pack(side="top")

# reset_button = customtkinter.CTkButton(text="restart", font=('consolas',20), command=new_game)
reset_button = Button(text="restart", font=('consolas',20), bg = 'grey', command=new_game)

reset_button.pack(side="top")

# frame = customtkinter.CTkFrame(window) # adding frame for the buttons to the window
frame = Frame(window) # adding frame for the buttons to the window

frame.pack()

for row in range(3):
    for column in range(3):
        # adding buttons to the frame and frame is added to window
        buttons[row][column] = Button(frame, text='', font=('consolas',40), bg='grey', width=5, height=2,
                                      command=lambda row=row, column=column: next_turn(row,column))
        buttons[row][column].grid(row=row,column=column)

window.mainloop()