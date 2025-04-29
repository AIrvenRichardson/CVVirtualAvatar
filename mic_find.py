import pygame
from pygame._sdl2.audio import get_audio_device_names

# THIS CODE IS INTENDED TO HELP YOU FIND THE CORRECT MIC FOR main.py
# CHANGE THE AUDIO_DEVICE_ID VARIABLE TO THE NUMBER NEXT TO THE MIC YOU WANT TO USE IN main.py

pygame.init()
names = get_audio_device_names(True)

print("This is a list of your audio devices:")
for i in range(len(names)):
    print(str(i) + ": "+ names[i])