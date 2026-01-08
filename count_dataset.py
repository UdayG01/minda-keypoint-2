import os

directory = r'c:\Work\Renata\Minda KeyPoint\minda_keypoint_detection_yolov11\new_dataset\dataset'

files = os.listdir(directory)

images = []
jsons = []

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

for f in files:
    name, ext = os.path.splitext(f)
    if ext.lower() in image_extensions:
        images.append(name)
    elif ext.lower() == '.json':
        jsons.append(name)

image_set = set(images)
json_set = set(jsons)

images_without_json = image_set - json_set

print(f"Total images: {len(images)}")
print(f"Total json files: {len(jsons)}")
print(f"Images without json: {len(images_without_json)}")
# print(f"List of images without json: {list(images_without_json)}")
