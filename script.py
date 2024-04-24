from tqdm import tqdm
import json
from googletrans import Translator
import requests
import os
import time

file_path = '/Users/purumalhotra/Downloads/DLProject/Dataset/instructions.json'
translator = Translator()
save_folder = '/Users/purumalhotra/Downloads/DLProject/Dataset/original_images'
json_path = '/Users/purumalhotra/Downloads/DLProject/Dataset/instructions.json'

if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
# Read the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

#print(data[0]['descriptions'])
for i in range(len(data)):
    imageFileName = 'u' + str(i+1) + '.png'
    data[i]['imageFileName'] = imageFileName
    
    for j in range(len(data[i]['descriptions'])):
        max_retries = 5
        retries = 0
        print(data[i]['descriptions'][j])
        while retries < max_retries:
            try:        
                translation = translator.translate(data[i]['descriptions'][j], src='de', dest='en').text
                data[i]['descriptions'][j] = str(translation)
                break
            except Exception as e:
                retries += 1
                time.sleep(5)
        
    save_path = os.path.join(save_folder, imageFileName)

    response = requests.get(data[i]['url'])
    response.raise_for_status()
    
    with open(save_path, 'wb') as file:
        file.write(response.content)

with open(json_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)