
src_path = "yolov5_trimmed_qat_precision_config_layer_arg.txt"
dst_path = "yolov5_trimmed_qat_precision_config_layer_arg_dst.txt"

# Step 1: Reading the .txt file content
with open(src_path, "r") as txt_file:
    content = txt_file.read().strip()

# Step 2: Splitting the content by comma to get individual items
items = content.split(',')[:-1]

# Step 3: Adding double quotes to each node in each item
modified_items = []
for item in items:
    # Splitting each item at the ':' to separate the node from the precision
    node, precision = item.split(':')
    # Adding double quotes around the node
    modified_item = f'"{node}":{precision}'
    modified_items.append(modified_item)

# Step 4: Joining the modified items back to a single string
modified_content = ",".join(modified_items)

# Step 5: Saving the modified content to a new .txt file
with open(dst_path, "w") as output_file:
    output_file.write(modified_content)
