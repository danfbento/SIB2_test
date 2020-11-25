from azureml.core.workspace import Workspace
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.webservice import AksWebservice
from azureml.core.model import InferenceConfig, Model
from azureml.core.environment import Environment, DEFAULT_GPU_IMAGE

# Initialize a workspace
ws = Workspace.from_config("C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/dev/.azureml/config.json")
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group,
      'Workspace connected', sep='\n')

# Choose a name for your cluster
aks_name = "SIB2-AKS-GPU"

# Check to see if the cluster already exists and create it if non existant
try:
    aks_target = ComputeTarget(workspace=ws, name=aks_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    # Provision AKS cluster with GPU machine
    prov_config = AksCompute.provisioning_configuration(vm_size="Standard_NC6")

    # Create the cluster
    aks_target = ComputeTarget.create(
        workspace=ws, name=aks_name, provisioning_configuration=prov_config
    )

    aks_target.wait_for_completion(show_output=True)

# Define the deployment configuration
gpu_aks_config = AksWebservice.deploy_configuration(autoscale_enabled=False,
                                                    num_replicas=3,
                                                    cpu_cores=2,
                                                    memory_gb=4)

# Define the inference configuration
myenv = Environment.from_conda_specification(name="testEnv",
                                             file_path="C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/deploy_env.yaml")

myenv.docker.base_image = DEFAULT_GPU_IMAGE
inference_config = InferenceConfig(#entry_script=os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'yolov5','score.py'),
                                   #entry_script="./yolov5/score.py",
                                   entry_script="score.py",
                                   environment=myenv,
                                   source_directory="C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/deployassets")

# Name of the web service that is deployed
aks_service_name = 'aks-mod5-test'
# Get the registerd model
model = Model(ws, "mod5_test", version=6)
# Deploy the model
aks_service = Model.deploy(ws,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=gpu_aks_config,
                           deployment_target=aks_target,
                           name=aks_service_name)

aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)

################################## troubleshooting
print(aks_service.get_logs())
aks_service.delete()

################################## test consumption

import base64
import json
import cv2
import requests
import os

img_path = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
#link = 'http://51.11.39.149:80/api/v1/service/aks-mod5-test/score'
link = aks_service.scoring_uri
#api_key = 'BLijQzGa9KcdG4IObGLQqdthXVHsnIqb'
api_key = aks_service.get_keys()[0]

def send2score(img_path, score_url, api_key):
    headers = {'Content-Type': 'application/json',
               'Authorization': ('Bearer ' + api_key)
               }
    
    img = cv2.imread(img_path)
    string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
    dict = {
        'imgname': os.path.basename(img_path),
        'img': string
        }
    jsonimg2 = json.dumps(dict, ensure_ascii=False, indent=4)
    resp = requests.post(url=link, data=jsonimg2, headers=headers)
    print(resp.text)

img_path2 = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x416_y0.png'
send2score(img_path=img_path2, score_url=link, api_key=api_key)

newimg = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x3744_y4160.png'
send2score(img_path=newimg, score_url=link, api_key=api_key)


















# Consumption (old attempts)

# 1
import requests
link = aks_service.scoring_uri
datadir = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo'
img = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
data = open(img, 'rb').read()

api_key = aks_service.get_keys()[0]
headers = {'Content-Type': 'application/json',
           'Authorization': ('Bearer ' + api_key)}

detection = requests.post(url=link, data=data, headers=headers) 
print(detection.text)

# 2
import json
result = aks_service.run(input_data=json.dumps({"url": img}))
print(aks_service.get_logs())

# 3
import os
url = aks_service.scoring_uri
path_img= 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
api_key = aks_service.get_keys()[0]
headers = {'Content-Type': 'application/json',
           'Authorization': ('Bearer ' + api_key)}

with open(path_img, 'rb') as img:
  name_img= os.path.basename(path_img)
  files= {'image': (name_img,img,'multipart/form-data',{'Expires': '0'}) }
  with requests.Session() as s:
    r = s.post(url,files=files,headers=headers)
    print(r.status_code)
    print(r.text)

# 4
import base64
import json
import cv2

img_path = 'C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test_geo/niuk_nie1_wv04_05042018_x832_y4992.png'
img = cv2.imread(img_path)
string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
dict_img = {
    'img': string
}
result = aks_service.run(input_data=json.dumps(dict_img))
detection = requests.post(url=link, data=dict_img, headers=headers) 
print(detection.text)

# 5
# client.py
import base64
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import cv2

img = cv2.imread(img_path)
img_b64 = base64.b64encode(img)
data = {'image': img_b64, 'shape': img.shape}
data = urlencode(data).encode("utf-8")
req = Request(link, data, headers=headers)
response = urlopen(req)
print(response.read().decode('utf-8'))

# 6
img = cv2.imread(img_path)
img_b64 = base64.b64encode(img)
data = {'image': img_b64, 'shape': img.shape}
data = urlencode(data).encode("utf-8")
input_data = json.dumps(data)
resp = requests.post(url=link, data=input_data, headers=headers)
print(resp.text)

# 7
img_file = open(img_path, "r")
#img_file.close()
# read the image file
data = img_file.read()        

# build JSON object
outjson = {}
outjson['img'] = data.encode('base64')   # data has to be encoded base64 and decode back in the Android app base64 as well
outjson['leaf'] = "leaf"
json_data = json.dumps(outjson)

# close file pointer and send data
img_file.close()
self.request.sendall(json_data)

# 8 20201030 = apparently img was sent and the load failed. Adapt datasets.py to decode
data = {}
with open(img_path, mode='rb') as imgfile:
    img = imgfile.read()
data['img'] = base64.encodebytes(img).decode("utf-8")

jsonimg = json.dumps(data)
resp = requests.post(url=link, data=jsonimg, headers=headers)
print(resp.text)

# 9 = this works! DB 20201102
# encode
img = cv2.imread(img_path)
string = base64.b64encode(cv2.imencode('.png', img)[1]).decode()
dict = {
    'img': string
}
jsonimg2 = json.dumps(dict, ensure_ascii=False, indent=4)
resp = requests.post(url=link, data=jsonimg2, headers=headers)
print(resp.text)

# troubleshooting
print(aks_service.get_logs())
aks_service.delete()