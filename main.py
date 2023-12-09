from train import train_model
from test import test_model
from CustomCellImage import test_model

import os

if __name__ == "__main__":
    mode = "test"  # Set to "train" or "test" here
    
    if mode == "train":
        num_epochs = 100 #10 default
        learning_rate = 0.0001 #0.0001 default
        batch_size = 64 #8 default
        target_size = (256,256) #(256,256) default
        train_model(num_epochs=num_epochs, learning_rate=learning_rate, batch_size=batch_size, target_size=target_size)
        
    elif mode == "test":
        test_model()

    elif mode == "custom_test":
        image_path = os.getcwd() + "\\Resources\\test\\011_img.png"
        #image_path = "C:\\Users\\chido\\Desktop\\600_100_RDG_2_1_20x-Actin.png"
        test_model(image_path)
    
    elif mode == "showmaskimages":
        from PIL import Image
        #show the images for *_msk.png under Resources/test
        for filename in os.listdir(os.getcwd() + "\\Resources\\test"):
            if filename.endswith("_masks.png"):
                print(filename)
                image = Image.open(os.getcwd() + "\\Resources\\test\\" + filename)
                #image.show()
                #image.close()
                #save image and convert to black and white scale
                image.save(os.getcwd() + "\\Resources\\test\\" + filename)
                continue
            else:
                continue
            
    