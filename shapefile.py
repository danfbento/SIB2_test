import os
import glob
import gdal
import subprocess
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

txt_dir = "C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/inference/output"
img_dir = "C:/Users/Danilo.Bento/Desktop/delete"
tiles_dir = "C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test/test_geo"

def tile_img(img_dir, tiles_dir):
    '''
    Create tiles of defined size
    '''
    img_dir = img_dir + '/'
    tiles_dir = tiles_dir + '/'
    for img in glob.glob(os.path.join(img_dir,"*.png")):
        tile_size_x = 416
        tile_size_y = 416
        
        input_filename = os.path.basename(img)
        output_filename = os.path.splitext(input_filename)[0]
        
        ds = gdal.Open(img)
        band = ds.GetRasterBand(1)
        xsize = band.XSize
        ysize = band.YSize
        
        for i in range(0, xsize, tile_size_x):
            for j in range(0, ysize, tile_size_y):
                #com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(img_dir) + str(input_filename) + " " + str(tiles_dir) + str(output_filename) + str(i) + "_" + str(j) + ".png"
                com_string = "gdal_translate -of GTIFF -srcwin " + str(i)+ ", " + str(j) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + str(img_dir) + str(input_filename) + " " + str(tiles_dir) + str(output_filename) + "_x" + str(i) + "_y" + str(j) + ".png"
                
                print(str(output_filename) + "_x" + str(i) + "_y" + str(j) + ".png")
                subprocess.call(com_string, shell=True)
                #os.system(com_string)
        

tile_img(img_dir=img_dir, tiles_dir=tiles_dir)

#############################################################
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

df = txt2csv(txt_dir=txt_dir)

#############################################################
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
        
        #x = tab['top_y'] # could be inverted
        #y = tab['left_x'] # could be inverted
        #x = tab['left_x']
        #y = tab['top_y']
        x = data_frame['xmin'] + (data_frame['height']/2)
        y = data_frame['ymin'] + (data_frame['width']/2)
        
        longitude = a * x + b * y + a * 0.5 + b * 0.5 + xoff
        latitude = d * x + e * y + d * 0.5 + e * 0.5 + yoff
        
        #print(shit[index])
        data_frame.at[index, 'pt_longitude'] = longitude[index]
        data_frame.at[index, 'pt_latitude'] = latitude[index]
        
        #data_frame['pt_longitude'] = a * x + b * y + a * 0.5 + b * 0.5 + xoff
        #data_frame['pt_latitude'] = d * x + e * y + d * 0.5 + e * 0.5 + yoff
    
    #data_frame.to_csv(csv, sep=',', encoding='utf-8', index=False)
    return data_frame        

df = px2xy(data_frame=df, tiles_dir=tiles_dir)

#############################################################
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
    
    #shp.plot(marker='*', markersize=0.2)
    
    return shp

shp = shapefile(data_frame=df)

shp_file = "C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test/test_geo/shp/detection.shp"
csv_file = "C:/Users/Danilo.Bento/Icon Dropbox/DEVDATA/RO/DEVELOPMENT/SIB2/tutorials/model5/mod5_deploy/test/test_geo/shp/detection.csv"
shp.to_file(shp_file, driver='ESRI Shapefile')
df.to_csv(csv_file, sep=',', encoding='utf-8', index=False)