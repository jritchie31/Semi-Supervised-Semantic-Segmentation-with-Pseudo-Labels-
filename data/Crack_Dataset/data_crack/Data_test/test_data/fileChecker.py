import os
import os.path as osp
import shutil
import filecmp

# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
dir1 = osp.join(current_dir, r"Unlabeled_Range_crop")
dir2 = osp.join(current_dir, r"labeled_Segmentation_rough")

# Get list of files in directory 1
files1 = set(os.listdir(dir1))

# Get list of files in directory 2
files2 = set(os.listdir(dir2))

# Find files with same name in both directories
common_files = files1.intersection(files2)

# Print the list of common files
print(common_files)


output_file = osp.join(current_dir, r"unlabeled.txt")
# Get list of files in dir1, dir2

# Open output file for writing
with open(output_file, 'w') as f:
    # Write each filename to the output file, with full path
    for filename in common_files:
        filepath = os.path.join("Crack_Dataset/data_crack/Data_train/split_1/Unlabeled_Range_crop", filename)
        f.write(filepath + '\n')



# Get the absolute path of the current file
current_dir = osp.dirname(osp.abspath(__file__))
dir1 = osp.join(current_dir, r"Unlabeled_Range_crop")
dir2 = osp.join(current_dir, r"labeled_Segmentation_crop")
dir3 = osp.join(current_dir, r"labeled_Range_crop")

# Get list of files in directory 1
files1 = set(os.listdir(dir1))

# Get list of files in directory 2
files2 = set(os.listdir(dir2))

# Find files with same name in both directories
common_files = files1.intersection(files2)

# Print the list of common files
print(common_files)

# Copy common files from directory 1 to directory 3
for file in common_files:
    src_path = os.path.join(dir1, file)
    dst_path = os.path.join(dir3, file)
    shutil.copy(src_path, dst_path)
    os.remove(src_path)

# Define list of files to ignore
ignore_list = ['.DS_Store']

# Compare files in directory 2 and directory 3, ignoring .DS_Store files
dir_cmp = filecmp.dircmp(dir2, dir3, ignore=ignore_list)
# Print the result of the comparison
if len(dir_cmp.left_only) == 0 and len(dir_cmp.right_only) == 0:
    print("The contents of directory 2 and directory 3 are the same.")
else:
    print("The contents of directory 2 and directory 3 are different.")
