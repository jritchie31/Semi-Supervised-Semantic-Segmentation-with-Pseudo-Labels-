# Open the file for reading and writing
with open('/Users/think327/Desktop/王灏林/GitHub/CEE8813-23S/U2PL-8813/results_report_0.5_Active_Semi/annotation_queue.txt', 'r+') as file:
    # Read the contents of the file
    contents = file.read()

    # Replace the specified string with an empty string
    contents = contents.replace('/Users/think327/Desktop/王灏林/GitHub/CEE8813-23S/U2PL-8813/data/', '')

    # Go back to the beginning of the file and write the modified contents
    file.seek(0)
    file.write(contents)
    file.truncate()
