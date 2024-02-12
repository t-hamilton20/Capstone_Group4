annotations = 'data/Complete/eight_x/sign_annotation.txt'
signs = 'data/Complete/eight_x/labels.txt'
new_annotations = 'data/Complete/eight_x/new_sign_annotation.txt'
new_signs = 'data/Complete/eight_x/new_labels.txt'
num_classes_to_remove = 50

# Define dictionaries to store class counts and class names
class_counts = {}
class_names = {}

# Read class names from the second file
with open(signs, 'r') as names_file:
    for line in names_file:
        parts = line.strip().split(': ')
        if len(parts) == 2:
            class_names[parts[0]] = parts[1]

total_instances = 0

# Open the main text file
with open(annotations, 'r') as file:
    # Read each line in the file
    for line in file:
        # Extract the class number from the line
        class_number = line.split('Class: ')[1].split(' ')[0]
        
        # Increment the count for the class in the dictionary
        class_counts[class_number] = class_counts.get(class_number, 0) + 1        
        total_instances += 1

# Sort the dictionary items by count in descending order
sorted_class_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

# Identify the 50 classes to be removed
classes_to_remove = [class_number for class_number, _ in sorted_class_counts[-num_classes_to_remove:]]

# Create a new dictionary excluding the 50 classes to be removed
filtered_class_names = {class_number: class_names[class_number] for class_number in class_names if class_number not in classes_to_remove}

# Create a mapping from old class numbers to new class numbers
class_mapping = {old_class: new_class for new_class, old_class in enumerate(filtered_class_names.keys())}

# Create a new annotations file and save class names
with open(new_annotations, 'w') as new_file:
    with open(annotations, 'r') as file:
        for line in file:
            class_number = line.split('Class: ')[1].split(' ')[0]
            if class_number not in classes_to_remove:
                new_class_number = class_mapping[class_number]
                new_file.write(line.replace(f'Class: {class_number}', f'Class: {new_class_number}'))

# Create a new labels file
with open(new_signs, 'w') as new_signs_file:
    for new_class_number, class_name in enumerate(filtered_class_names.values()):
        new_signs_file.write(f'{new_class_number}: {class_name}\n')

# Print information about the remaining classes
print(f"Remaining classes after removing the lowest occurring {num_classes_to_remove} classes:")
for new_class_number, class_name in enumerate(filtered_class_names.values()):
    print(f'Class {new_class_number}: {class_name}')

print(f"Total number of remaining classes: {len(filtered_class_names)}")
