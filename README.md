This is my virtual avatar camera application, it uses a simpleCNN model trained on 3 classes of facial affect (using only images, not facial points, which likely would be more accurate) along with your camera and microphone to make a simple virtual avatar act. <br>

This project was really just intended for me to work with some computer vision, but you're more than welcome to use it, to do so:
1. PLEASE make a venv (Python 3.11.11 and 3.13.3 work, anything after or in-between will probably work as well).
2. pip install -r requirements.txt while in the root
3. Run main.py

There are two things you may need to modify in main.py, they are both declared as global variables near the top (lines 6 and 7).<br>
    TALKING_MIC_LEVEL determines how loud the mic input needs to be to trigger the talking state<br>
    AUDIO_DEVICE_ID is your microphone id, this is likely 0 or 1 for you, if you're unsure you can run mic_find.py<br>

CONTROLS <br>
    l - (L as in lock) locks the avatar to the center of the screen<br>

CHANGING AVATARS<br>
    The avatarimgs directory should have a folder under it for each avatar (see: ghost and cat).<br>
    If you want to make your own, simply make a folder with the avatar name and add a folder under it titled "talking" (without quotation marks)<br>
    Then add a neu.png, hap.png, and an ang.png under both the avatar folder and the talking folder with your desired looks.<br>
    It should look like:

    avatarimgs/
        ...
        Custom/
            Talking/
                ang.png
                hap.png
                neu.png
            ang.png
            hap.png
            neu.png
        ...
Then simply change the AVATAR variable to be the avatar's base folder (in this case, "Custom")

EXPRESSIONS
    The model is not great, it only has pretty exaggerated images of expressions to work with. For the most part, any big smile (especially with teeth) will trigger happy, lowering your brow or doing a Kubrick stare will trigger angry, and anything else will be neutral.

Thanks for checking out my project!
