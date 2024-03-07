'''
Counts the number of occurances for each class given the annotations and labels files
Commented out code compares against another annotation and labels file to see if they match
'''

def count_classes(annotations_file, class_names_file):

    # Define dictionaries to store class counts and class names
    class_counts = {}
    class_names = {}

    # Read class names from the second file
    with open(class_names_file, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                class_names[parts[0]] = parts[1]

    total_instances = 0

    # Open the main text file
    with open(annotations_file, 'r') as file:
        # Read each line in the file
        for line in file:
            # Extract the class number from the line
            class_number = line.split('Class: ')[1].split(' ')[0]
        
            # Increment the count for the class in the dictionary
            class_counts[class_number] = class_counts.get(class_number, 0) + 1        
            total_instances += 1


    # Sort the dictionary items by count in descending order
    sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    # Print the sorted class counts with class names
    for class_number, count in sorted_class_counts:
        class_name = class_names.get(class_number, f'Unknown Class {class_number}')
        proportion = (count / total_instances) * 100
        print(f'Class {class_number}: {class_name} - {count} occurrences - {proportion:.2f}% of total')

    print(f"Total number of classes: {len(class_counts)}")
    return class_names 

def compare_classes(class_names_file, class_names_to_compare):
    class_names = {}

    # Read class names from the second file
    with open(class_names_file, 'r') as names_file:
        for line in names_file:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                class_names[parts[0]] = parts[1]

    for key in class_names:
        val1 = class_names_to_compare[key]
        val2 = class_names[key]
        if val1 != val2:
            print(f"MISMATCH: {key}, {val1} and {val2}")

        else:
            print("matching")

annotations_file = 'data/Complete/annotations.txt'
class_names_file = 'data/class_names.txt'
class_names = count_classes(annotations_file, class_names_file)
val_classes_file = 'data/val_class_names.txt'
compare_classes(val_classes_file, class_names)
