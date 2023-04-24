import os
import os.path as osp

# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
dir1 = osp.join(current_dir, r"Unlabeled_Range_crop")
dir2 = osp.join(current_dir, r"labeled_Segmentation_crop")
output_file1 = osp.join(current_dir, r"unlabeled.txt")
output_file2 = osp.join(current_dir, r"labeled.txt")
# Get list of files in dir1, dir2
file_list1 = os.listdir(dir1)
file_list2 = os.listdir(dir2)

# Open output file for writing
with open(output_file1, 'w') as f:
    # Write each filename to the output file, with full path
    for filename in file_list1:
        filepath = os.path.join("Crack_Dataset/data_crack/Data_train/split_1/Unlabeled_Range_crop", filename)
        f.write(filepath + '\n')
# Open output file for writing
with open(output_file2, 'w') as f:
    # Write each filename to the output file, with full path
    for filename in file_list2:
        filepath = os.path.join("Crack_Dataset/data_crack/Data_train/split_1/labeled_Segmentation_crop", filename)
        f.write(filepath + '\n')
