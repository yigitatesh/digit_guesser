import pygame
import sys, os
import cv2
import numpy as np

from keras.models import load_model

from button import Button, Frame

#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Paint(object):
    """Runs paint"""
    ## Constants
    WIDTH = 1000
    HEIGHT = 700
    FPS = 60
    ## Colors
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    RED = (255,0,0)
    GREEN = (0,255,0)
    BLUE = (0,0,255)
    GRAY = (127,127,127)
    
    YELLOW = (255,255,0)
    MAGENTA = (255,0,255)
    CYAN = (0,255,255)

    LILAC = (200,162,200)
    VIOLET = (238,130,238)

    CREAM = (255,253,208)

    AQUAMARINE = (127,255,212)
    LIGHTAQUAMARINE = (147, 255, 232)
    TURQUOISE = (67, 198, 219)
    LIGHTSLATE = (204, 255, 255)
    TEAL = (0, 128, 128)

    LIGHTGRAY = (200,200,200)
    PLATINUM = (229, 228, 226)

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Paint")
        self.clock = pygame.time.Clock()

        ## Fonts
        self.smallfont = pygame.font.SysFont("comicsansms", 25)
        self.medfont = pygame.font.SysFont("comicsansms", 45)
        self.largefont = pygame.font.SysFont("comicsansms", 80)

        # neural network
        self.cnn = load_model("data/digit_model.h5")
        self.preprocessed_images = []

        # image shape for digit prediction
        self.imshape = (28, 28)

    ## Helper Functions ##
    def exit_program(self):
        """terminate the program"""
        pygame.quit()
        sys.exit(0)
        #os._exit(0)

    def text_to_screen(self, x, y, text, color, font):
        """put text to screen (x and y is center of the text)"""
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        text_rect.center = (x, y)
        self.screen.blit(text_surf, text_rect)
    
    def text_to_screen_left(self, x, y, text, color, font):
        """put text to screen (x and y is center of the text)"""
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        text_rect.left = x
        text_rect.top = y
        self.screen.blit(text_surf, text_rect)
    
    def roundline(self, color, start, end):
        if end is None:
            pass
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            pygame.draw.circle(self.screen, color, (x, y), self.draw_radius)
    
    def clear(self):
        """Clear the canvas"""
        self.screen.fill(self.canvas_color)
        self.is_predicted = False
    
    ## Image Functions
    def preprocess_image(self):
        """Preprocess screen image for contour finding.
        Each contour would be a digit."""

        self.is_predicted = True
        # gets 3d image array of canvas
        self.raw_image = pygame.surfarray.array3d(self.screen)
        self.raw_image = self.raw_image.swapaxes(0, 1)
        self.raw_image = self.raw_image[self.canvas.top:self.canvas.bottom,
                                        self.canvas.left:self.canvas.right]

        # converts it into grayscale
        self.processed_image = cv2.cvtColor(self.raw_image, cv2.COLOR_RGB2GRAY)

        # binarize with thresh
        ret, self.processed_image = cv2.threshold(self.processed_image.copy(), 100, 255, cv2.THRESH_BINARY_INV)

        return self.processed_image

    def find_contour_boxes(self, image):
        """Finds contour boxes in a binarized image"""
        contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [] # each box is [x, y, w, h]
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            
            boxes.append([x, y, w, h])
            
        return boxes

    def process_digit_contour(self, digit):
        """Process digit contour to prepare for neural networks"""

        # resize digit to 18x18
        resized = cv2.resize(digit, (18, 18), interpolation=cv2.INTER_AREA)

        # add padding to side for creating some space
        padded = cv2.copyMakeBorder(resized, 5, 5, 5, 5, cv2.BORDER_CONSTANT, 0)
        
        # scale images
        padded = padded / 255.0

        # negate image
        padded = 1 - padded
        
        return padded

    def predict_from_boxes(self, model, image, boxes):
        """Predict digits for each digit contour"""
        preds, confs, digit_boxes = [], [], []
        
        for x, y, w, h in boxes:
            # width and height threshold
            if w < 20 or h < 20:
                continue

            # crop digit
            digit = image[y:y+h, x:x+w]

            # process digit contour
            processed = self.process_digit_contour(digit)

            #cv2.imshow("processed digit", processed)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            # prediction
            probs = np.squeeze(model.predict(processed.reshape(1, 28, 28, 1)))
            pred = probs.argmax()
            conf = probs.max()

            # prediction threshold
            if conf < 0.25:
                continue

            #print(pred, conf)
            
            preds.append(pred)
            confs.append(conf)
            digit_boxes.append([x, y, w, h])

        return preds, confs, digit_boxes
    
    def predict_show(self):
        """Preprocess and predict digits in a raw image.
        Then show predictions drawn as boxes with confidence levels"""

        # preprocess image
        processed_image = self.preprocess_image()

        # find contour boxes
        boxes = self.find_contour_boxes(processed_image)

        # predict from boxes
        preds, confs, digit_boxes = self.predict_from_boxes(self.cnn, processed_image, boxes)

        # cv2 showing image with predictions
        show_image = self.raw_image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(preds)):
            (x, y, w, h) = digit_boxes[i]
            x2 = x + w
            y2 = y + h

            # rectangle
            cv2.rectangle(show_image, (x, y), (x2, y2), self.GREEN, 5)

            # text
            cv2.putText(show_image, "{}, %{:2f}".format(preds[i], confs[i]*100),
                (x - 5, y - 5), font, 3/4, self.RED, 2, cv2.LINE_AA)

        cv2.imshow("image", show_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ## PAGES ##
    def paint_page(self):
        """the page that paint runs"""
        # image
        self.prediction = None
        self.proba_prediction = None
        self.is_predicted = False

        # screen, frames and canvas
        self.screen_rect = self.screen.get_rect()
        self.background_color = self.CREAM

        self.right_frame_margin = 50
        self.top_frame_color = self.LIGHTGRAY
        self.top_frame_width = 50
        self.top_frame = pygame.Rect(0, 0, self.WIDTH, self.top_frame_width)
        self.right_frame_width = 300
        self.right_frame = pygame.Rect(self.WIDTH - self.right_frame_width, 
                                        0, self.right_frame_width, self.HEIGHT)
        self.canvas = pygame.Rect(0, self.top_frame_width, 
                                    self.WIDTH - self.right_frame_width, self.HEIGHT)
        self.canvas_color = self.WHITE

        # drawing
        self.drawing = False
        self.draw_color = self.BLACK
        self.draw_radius = 5
        self.mouse_pos = None
        self.last_pos = None
        self.screen.fill(self.canvas_color)

        #self.main_frame = Frame(self.screen, self.BLUE, self.screen_rect, 10)

        # Buttons
        self.predict_button_x = self.right_frame.left + self.right_frame_margin
        self.predict_button_y = self.top_frame.bottom + self.right_frame_margin
        self.predict_button_width = 140
        self.predict_button_height = 50
        self.predict_button = Button(self.screen, self.predict_button_x, self.predict_button_y,
                            self.predict_button_width, self.predict_button_height, self.BLACK,
                            self.LIGHTAQUAMARINE, self.AQUAMARINE, self.LIGHTAQUAMARINE, 
                            action=lambda: self.predict_show(), text="Tahmin et", font=self.smallfont)
        
        self.clear_button_width = 140
        self.clear_button_height = 50
        self.clear_button_x = self.right_frame.left + self.right_frame_margin
        self.clear_button_y = self.HEIGHT - self.right_frame_margin - self.clear_button_height
        self.clear_button = Button(self.screen, self.clear_button_x, self.clear_button_y,
                                self.clear_button_width, self.clear_button_height, self.BLACK,
                                self.LIGHTAQUAMARINE, self.AQUAMARINE, self.LIGHTAQUAMARINE,
                                action=lambda: self.clear(), text="Temizle", font=self.smallfont)

        self.buttons = [self.predict_button, self.clear_button]

        run = True
        while run:
            ## fps
            self.clock.tick(self.FPS)

            ## process inputs
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                # check for inputs
                # mouse position
                self.mouse_pos = pygame.mouse.get_pos()
                if event.type == pygame.MOUSEMOTION:
                #   #self.mouse_pos = pygame.mouse.get_pos()
                    if self.drawing:
                        if self.canvas.collidepoint(self.mouse_pos):
                            pygame.draw.circle(self.screen, self.draw_color, 
                                                self.mouse_pos, self.draw_radius)
                            self.roundline(self.draw_color, self.mouse_pos, self.last_pos)
                    self.last_pos = self.mouse_pos

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.canvas.collidepoint(self.mouse_pos):
                        self.drawing = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.drawing = False

            ## update
            for button in self.buttons:
                button.update()

            ## draw
            # right frame
            pygame.draw.rect(self.screen, self.background_color, self.right_frame)
            # top frame
            pygame.draw.rect(self.screen, self.top_frame_color, self.top_frame)
            #self.main_frame.draw()

            # draw circle
            if self.drawing and self.canvas.collidepoint(self.mouse_pos):
                pygame.draw.circle(self.screen, self.draw_color, 
                                        self.mouse_pos, self.draw_radius)
            
            for button in self.buttons:
                button.draw()

            # flip screen after drawing
            pygame.display.flip()

        self.exit_program()


if __name__ == "__main__":
    paint = Paint()
    paint.paint_page()