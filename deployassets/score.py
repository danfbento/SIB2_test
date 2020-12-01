
import os
import torch
import shutil
from numpy import random
import time
from pathlib import Path
import cv2
import mysql.connector
from mysql.connector import errorcode

# Custom imports
from deployassets.models.experimental import attempt_load
from deployassets.utils.torch_utils import select_device, load_classifier, time_synchronized
from deployassets.utils.general import (check_img_size, set_logging, non_max_suppression, 
                           apply_classifier, scale_coords, xyxy2xywh, plot_one_box)
from deployassets.utils.datasets import LoadImages

################################ CUSTOM FUNCTIONS
def return_results(results_dir):
    all_files = os.listdir(os.path.abspath(results_dir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
    
    summary = []
    for txt in data_files:
        with open(os.path.join(results_dir,txt)) as f:
            for line in f:
                content = line.split()
                
                summary.append({
                    'image': txt,
                    'detected_class': content[0],
                    'detected_coord1': content[1],
                    'detected_coord2': content[2],
                    'detected_coord3': content[3],
                    'detected_coord4': content[4]
                })
    
    # Obtain connection string information from the portal
    config = {
      'host':'sib2-mysql-server.mysql.database.azure.com',
      'user':'icon_sib2@sib2-mysql-server',
      'password':'D02_YK29',
      'database':'SIB2DB',
      #'client_flags': [ClientFlag.SSL],
      'ssl_ca': str(os.path.join('.', 'deployassets', 'ssl', 'DigiCertGlobalRootG2.crt.pem'))
      #'ssl_verify_cert':'true'
    }
    
    # Construct connection string
    try:
      conn = mysql.connector.connect(**config)
      print("Connection established")
    except mysql.connector.Error as err:
      if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with the user name or password")
      elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
      else:
        print(err)
    else:
      cursor = conn.cursor()
      
      counter = 1
      total = len(summary)    
      for dictionary in summary:
        # Insert data into table
        cursor.execute("INSERT INTO model5_test_detections (image, detected_class,detected_coord1, detected_coord2, detected_coord3, detected_coord4) VALUES (%s, %s, %s, %s, %s, %s);",
                      (dictionary['image'], 
                       dictionary['detected_class'], 
                       dictionary['detected_coord1'], 
                       dictionary['detected_coord2'], 
                       dictionary['detected_coord3'], 
                       dictionary['detected_coord4']
                       )
                      )
        print(f"Detection {counter} from {total}." 
              f"Detected class = {dictionary['detected_class']}")
        counter = counter +1
      print("Inserted",total,"row(s) of data into the database")
      
      # Cleanup
      conn.commit()
      cursor.close()
      conn.close()
      print("Done.")
        
################################ SCORE FUNCTIONS
def init():
    global model
    global device
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mod5_azure', 'yolov5','runs','exp0', 'weights', 'last.pt') # Azure path
    #model_path = 'runs/exp0/weights/mod5_test_weight.pt' # Colab path
    device=''
    device = select_device(device)
    
    print('Loading model...', end='')
    model = attempt_load(model_path, map_location=device)  # load FP32 model
    
    print('Loaded. Success!')
    
def run(input_images):
    # Arguments transformed in constants
    conf_thres = 0.4 # confidence threshold
    imgsz = 416 # image size
    iou_thres = 0.5 # IOU threshold for NMS
    agnostic_nms = False
    augment = False
    classes = None
    save_txt = True
    view_img=False
    # out (output location) is defined after set_logging()
    
    # Initialize
    set_logging()
    
    out = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'mod5_azure', 'yolov5','inference','output') # Azure path
    #out = 'inference/output' # Colab path
    
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # Load model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16
        
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
        
    # Set Dataloader
    # vid_path, vid_writer = None, None
    # if webcam:
    #     view_img = True
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz)
    # else:
    #     save_img = True
    #     dataset = LoadImages(source, img_size=imgsz)
    save_img = True
    dataset = LoadImages(input_images, img_size=imgsz)
    
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap, img0name in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()
        
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            # else:
            #     p, s, im0 = path, '', im0s
            p, s, im0 = path, '', im0s
        
            #save_path = str(Path(out) / Path(p).name) # original
            #save_path = str(Path(out) / 'test.png') # DB 20201102
            save_path = str(Path(out) / img0name) # DB 20201103
            txt_name = img0name.replace('.png', '.txt') # DB 20201103
            #txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '') # original
            txt_path = str(Path(out) / txt_name) + ('_%g' % dataset.frame if dataset.mode == 'video' else '') # DB 20201103
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                            
                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
                
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)                
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
                    
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
    
    #results = return_results(Path(out))
    return_results(Path(out))
    #return results

    print('Done. (%.3fs)' % (time.time() - t0))