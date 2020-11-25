import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

# customized imports
import gdal
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

##################################################### customized functions
def txt2csv(txt_dir):
    '''
    Get the txt files from detection and create a csv with them
    '''
    all_files = os.listdir(os.path.abspath(txt_dir))
    data_files = list(filter(lambda file: file.endswith('.txt'), all_files))
    final_list = []
    
    for txt in data_files:
        with open(os.path.join(txt_dir,txt), 'r') as txtfile:
            # Stripping data from the txt file into a list #
            list_of_lists = []
            for line in txtfile:
                stripped_line = line.strip()
                line_list = stripped_line.split()
                list_of_lists.append(line_list)
            
            # Conversion of str to int #
            stage1 = []
            for i in range(0, len(list_of_lists)):
                test_list = list(map(float, list_of_lists[i])) 
                stage1.append(test_list)

            # Denormalizing # 
            stage2 = []
            mul = [1,416,416,416,416] #[constant, image_width, image_height, image_width, image_height]
            for x in stage1:
                c,xx,yy,w,h = x[0]*mul[0], x[1]*mul[1], x[2]*mul[2], x[3]*mul[3], x[4]*mul[4]    
                stage2.append([c,xx,yy,w,h])

            # Convert (x_center, y_center, width, height) --> (x_min, y_min, width, height) #
            stage_final = []
            for x in stage2:
                img_name = txt
                c,xx,yy,w,h = x[0]*1, (x[1]-(x[3]/2)) , (x[2]-(x[4]/2)), x[3]*1, x[4]*1  
                stage_final.append([img_name,c,xx,yy,w,h])
        
        for el in stage_final:
            final_list.append({
                'image': el[0],
                'class': el[1],
                'xmin': el[2],
                'ymin': el[3],
                'width': el[4],
                'height': el[5]
            })
        
    final_df = pd.DataFrame(final_list)
    final_df['class'] = final_df['class'].astype(str)
    final_df['class'] = final_df['class'].replace(str(0.0),'cow')
    final_df['class'] = final_df['class'].replace(str(1.0),'sheep')
    final_df['class'] = final_df['class'].replace(str(2.0),'object')
    
    final_df['xmax'] = final_df['width'] + final_df['xmin']
    final_df['ymax'] = final_df['height'] + final_df['ymin']
    
    return final_df

#####################################################
def px2xy(data_frame, tiles_dir):
    '''
    Transform image xy to real world coordinates
    '''
    data_frame['pt_longitude'] = 0.0
    data_frame['pt_latitude'] = 0.0
    
    #for index in data_frame.index:
    #for row in data_frame.itertuples():
    for index, row in data_frame.iterrows():
        img = os.path.join(tiles_dir, row['image'])
        img = img.replace('.txt', '.png')
        print(row['image'])
        
        # https://stackoverflow.com/questions/50191648/gis-geotiff-gdal-python-how-to-get-coordinates-from-pixel
        #https://stackoverflow.com/questions/52443906/pixel-array-position-to-lat-long-gdal-python
        ds = gdal.Open(img)
        xoff, a, b, yoff, d, e = ds.GetGeoTransform()
        
        x = data_frame['xmin'] + (data_frame['height']/2)
        y = data_frame['ymin'] + (data_frame['width']/2)
        
        longitude = a * x + b * y + a * 0.5 + b * 0.5 + xoff
        latitude = d * x + e * y + d * 0.5 + e * 0.5 + yoff
        
        data_frame.at[index, 'pt_longitude'] = longitude[index]
        data_frame.at[index, 'pt_latitude'] = latitude[index]
        
    return data_frame

#####################################################
def shapefile(data_frame):
    '''
    Create a shapefile from the csv
    '''
    # combine lat and lon column to a shapely Point() object
    data_frame['geometry'] = data_frame.apply(
        lambda x: Point((float(x.pt_longitude), float(x.pt_latitude))
                        ), axis=1)
    
    crs = {'init': 'epsg:29902'}
    shp = gpd.GeoDataFrame(data_frame, crs=crs, geometry='geometry')
    
    return shp

##################################################### DETECTION
def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
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
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
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
        
        print('Starting geospatial operations')
        df = txt2csv(txt_dir=Path(out))
        df = px2xy(data_frame=df, tiles_dir=source)        
        
        spatial_folder = out + '/spatial'
        if os.path.exists(spatial_folder):
            shutil.rmtree(spatial_folder)  # delete spatial output folder
        os.makedirs(spatial_folder)  # make new spatial output folder
        
        csv_file = spatial_folder + '/detections.csv'
        df.to_csv(csv_file, sep=',', encoding='utf-8', index=False)
        
        shp = shapefile(data_frame=df)
        shp_file = spatial_folder + '/detections.shp'
        shp.to_file(shp_file, driver='ESRI Shapefile')

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
