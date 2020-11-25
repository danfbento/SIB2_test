import os

def return_results(results_dir):
    all_files = os.listdir(os.path.abspath(results_dir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
    
    summary = []
    for txt in data_files:
        with open(os.path.join(results_dir,txt)) as f:
            line_count = sum(1 for line in f if line.strip())
        #print(f"{line_count} animals detected in {txt}")
        summary.append([txt, line_count])
    
    #results = {item[0]: item[1] for item in summary}
    keys = ['image', 'animals']
    results = {x:list(y) for x,y in zip(keys, zip(*summary))}
    #print(results)
    print(results['image'][0],results['animals'][0])
    return results
    
        

results_dir = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/inference/output'
res_test = return_results(results_dir)

###########################################################
import base64
import json
import os
import cv2

img_path = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
img_original = cv2.imread(img_path)
print(type(img_original))

# 8 20201030 = apparently img was sent and the load failed. Adapt datasets.py to decode
data = {}
with open(img_path, mode='rb') as imgfile:
    img = imgfile.read()
data['img'] = base64.encodebytes(img).decode("utf-8")

jsonimg = json.dumps(data)
#resp = requests.post(url=link, data=jsonimg, headers=headers)
#print(resp.text)

# receive back
load = json.loads(jsonimg)
imdata = base64.b64decode(load['img'])
img_np = cv2.imdecode(imdata, cv2.CV_LOAD_IMAGE_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1

print(type(imdata))
img = cv2.imread(imdata)

os.path.isfile(imdata) # probably will never work

###########################################################
import base64
import json
import cv2
import requests

img_path = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
link = 'http://51.11.39.149:80/api/v1/service/aks-mod5-test/score'
api_key = 'bO7sp1iNprvKohAvB1rVtDcdOpHo9BXh'
headers = {'Content-Type': 'application/json',
           'Authorization': ('Bearer ' + api_key)
           }

# encode
img = cv2.imread(img_path)
string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
dict = {
    'img': string
}
jsonimg2 = json.dumps(dict, ensure_ascii=False, indent=4)
resp = requests.post(url=link, data=jsonimg2, headers=headers)
print(resp.text)
with open('C:/Users/Danilo.Bento/Desktop/temp/garbage/resp.txt', 'w') as outfile:
    outfile.write(resp.text)


with open('./0.json', 'w') as outfile:
    json.dump(dict, outfile, ensure_ascii=False, indent=4)

# dencode    
import base64
import json
import cv2
import numpy as np

response = json.loads(jsonimg2)
string = response['img']
jpg_original = base64.b64decode(string)
jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
img = cv2.imdecode(jpg_as_np, flags=1)
os.path.isfile(img) # probably will never work

cv2.imwrite('C:/Users/Danilo.Bento/Desktop/temp/garbage/0.png', img)

######################################################
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
files = [img_path]
images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
images




x = 'niuk_nie1_wv04_05042018_x832_y4992.png'
x = x.replace('.png', '.txt')