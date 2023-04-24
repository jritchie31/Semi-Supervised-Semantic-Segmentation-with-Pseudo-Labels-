# Open the file for reading and writing
with open('/Users/think327/Desktop/王灏林/GitHub/CEE8813-23S/U2PL-8813/data/Crack_Dataset/data_crack/Data_test/val_data/val_label.txt', 'r+') as file:
    # Read the contents of the file
    contents = file.read()

    # Replace all backslashes with forward slashes
    contents = contents.replace('\\', '/')
    contents = contents.replace('Segmentation_crop', 'Range_crop')
    # Go back to the beginning of the file and write the modified contents
    file.seek(0)
    file.write(contents)
    file.truncate()
