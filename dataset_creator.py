# Following function will take in the dataset and create necessary file paths for .txt to reference into
# .yaml file
import os


def convertFiles(onedrive_folder, path_images):
    # Set the path to your OneDrive folder

    # Set the path to your image folder within the OneDrive folder
    image_folder = os.path.join(onedrive_folder, path_images)
    # Get a list of all .png files in the image folder
    png_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".png")]

    # Create a text file and write the file paths to it
    with open("data/Crack_Dataset/data_crack/Data_test/test data/unlabeled.txt", "w") as f:
        for file_path in png_files:
            file_path = file_path.replace("C:/Users/Jayda Ritchie/OneDrive - Georgia Institute of Technology/", "")
            f.write(file_path + "\n")


folder = 'C:/Users/Jayda Ritchie/OneDrive - Georgia Institute of Technology'
images = "C:/Users/Jayda Ritchie/OneDrive - Georgia Institute of Technology/Crack_Dataset/data_crack/Data_test/test data/Unlabeled_Range_crop"
convertFiles(folder, images)
# labeled_Segmentation_crop"
#Unlabeled_Range_crop"
