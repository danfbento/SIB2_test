from azureml.core.workspace import Workspace
from azureml.core import Model

# Initialize a workspace
ws = Workspace.from_config("C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/dev/.azureml/config.json")
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group,
      'Workspace connected', sep='\n')

model = Model.register(workspace=ws,
                       model_name='mod5_test',
                       model_path='C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/mod5_azure', # local path
                       #child_paths=['C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/mod5_azure/yolov5/runs/exp0/weights/last.pt'],
                       description='Test model based on model 5 done in phase 1. It trained for 250 epochs. Contain original folder structure',
                       #tags={'dept': 'sales'},
                       model_framework=Model.Framework.PYTORCH,
                       model_framework_version='1.6.0')

for model in Model.list(ws):
    # Get model name and auto-generated version
    print(model.name, 'version:', model.version)
    
# import logging
# logging.basicConfig(level=logging.DEBUG)
# print(Model.get_model_path(model_name='mod5_test'))