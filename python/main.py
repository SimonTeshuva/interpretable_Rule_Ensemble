import os
import functions

def print_fucntion_names():
    current_path = os.getcwd()
    count = 1
    for file in os.listdir(current_path):
        if file != "main.py":
            print(str(count)+": " + "".join(file.split(".")[:-1]))
    print(current_path)

def import_script(pmName):
    pm = __import__(pmName)

def menu():

    return


if __name__ == "__main__":
    menu()