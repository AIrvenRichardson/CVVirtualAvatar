import cv2, torch, math, pygame, array, time, numpy as np, cv2.data as data
from torch import nn
from operator import indexOf
from pygame._sdl2.audio import get_audio_device_names, AudioDevice, AUDIO_F32, AUDIO_ALLOW_FORMAT_CHANGE

# VARIABLES YOU CAN/SHOULD CHANGE
TALKING_MIC_LEVEL = 0.009 # FIND A VALUE THAT WORKS FOR YOUR MIC (HYPERPARAMETER, YOU WILL LIKELY NEED TO ADJUST THIS, MY MIC SUCKS)
AUDIO_DEVICE_ID = 1 # Run mic_find.py to find this value for you (0 is your default mic)
AVATAR = "cat" # Change to the folder name of the avatar you want to use. Custom avatar instructions are in the readme

# Audio callback that modifies a global variable to control the avatar's talking state
global talking
def callback(audiodevice, audiomemoryview):
    #https://www.youtube.com/watch?v=675teI6-_-g RMS implementation source
    global talking
    sound_data = array.array('f')
    sound_data.frombytes(bytes(audiomemoryview))
    rms = 0
    for i in range(0, len(sound_data), 2):
        sample = sound_data[i] + sound_data[i+1]
        rms += sample*sample
    rms = math.sqrt(rms/512)
    if rms > TALKING_MIC_LEVEL: # FIND A VALUE THAT WORKS FOR YOUR MIC (HYPERPARAMETER, YOU WILL LIKELY NEED TO ADJUST THIS, MY MIC SUCKS)
        talking = True
    else:
        talking = False


def main():
    global talking
    # CAMERA SETUP
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    frame_width = int(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = int(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    haar_cascade = cv2.CascadeClassifier(data.haarcascades + 'haarcascade_frontalface_default.xml')

    # MODEL SETUP
    class_names = ['angry', 'happy', 'neutral']
    emodel = SimpleCNN(1,64,3)
    emodel.load_state_dict(torch.load("mod.pth"))
    emodel.eval()
    last_face = []
    average_emotion = [0,0,0]

    # WINDOW SETUP
    pygame.init()
    screen = pygame.display.set_mode((640,480)) #CV2 default window
    pygame.display.set_caption("Virtual Camera")
    clock = pygame.time.Clock()
    running = True

    # AVATAR SETUP
    basepath = 'avatarimgs/' + AVATAR + '/'
    avatarimgs = [pygame.image.load(basepath + 'ang.png'),
                  pygame.image.load(basepath + 'hap.png'),
                  pygame.image.load(basepath + 'neu.png'),
                  pygame.image.load(basepath + 'talking/ang.png'),
                  pygame.image.load(basepath + 'talking/hap.png'),
                  pygame.image.load(basepath + 'talking/neu.png')]
    follow = True
    headpos = [0,0]

    def avi(x,y,img):
        screen.blit(img, (x,y))

    # AUDIO SETUP
    names = get_audio_device_names(True)
    audio = AudioDevice(
        devicename=names[AUDIO_DEVICE_ID], # THIS DETERMINES WHAT MICROPHONE TO USE, IF IT DOESN'T WORK TRY A DIFFERENT NUMBER LIKE 0 OR 1
        iscapture=True,
        frequency=44100,
        audioformat=AUDIO_F32,
        numchannels=2,
        chunksize=512,
        allowed_changes=AUDIO_ALLOW_FORMAT_CHANGE,
        callback=callback,)
    audio.pause(0)

    while running:
        #Run at a framerate
        start = time.time()

        # COMPUTER VISION PART
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

        for (x, y, w, h) in faces_rect:
            last_face = gray[y:y+h, x:x+w] 
            face = cv2.resize(last_face, (48,48))
            face = face.astype(np.float32)/255.0
            face = torch.from_numpy(face)
            face = face.unsqueeze(dim=0)
            face = face.unsqueeze(dim=0)

            headpos[0] = (x + w/2)
            headpos[1] = (y + h/2)

            with torch.inference_mode():
                predicted = emodel(face)
                pred_probs = torch.softmax(predicted, dim=1)
                average_emotion[torch.argmax(pred_probs)] += 1

        if talking:
            img = avatarimgs[indexOf(average_emotion, max(average_emotion)) + 3]
        else:
            img = avatarimgs[indexOf(average_emotion, max(average_emotion))]
        img = pygame.transform.scale(img, (200, 350))
        average_emotion = [0,0,0]

        # WINDOW/PYGAME PART
        screen.fill("green")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    follow = not follow

        if follow:
            avi(headpos[0]-img.get_width()/2,headpos[1]-img.get_height()/2, img)
        else:
            avi(screen.get_width()/2 - img.get_width() / 2, screen.get_height()/2 - img.get_height() / 2, img)

        pygame.display.flip()

        clock.tick(30)
        time.sleep(max(1./30 - (time.time() - start), 0))


    cam.release()
    cv2.destroyAllWindows()
    pygame.quit()

#This is the model used for detection, having it here means I don't need to include training scripts.
class SimpleCNN(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 12 * 12,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

if __name__ == "__main__":
    main()