import ee 
import os
import pandas as pd
import geemap
import logging


class EePull:
    def __init__(self, log_dir: str, name: str):
        self.log_dir = log_dir
        self.name = name
        self.__set_logger(self.log_dir, self.name)

    def __set_logger(self, log_dir, name):
        logging.basicConfig(filename=os.path.join(log_dir,f"log_{name}.log"), level=logging.INFO, format = '%(asctime)s:%(levelname)s:%(message)s')    


    def export_landsat_eight_images(self, glims_id, bounding_box, start_date,end_date, out_dir):
        
        region = ee.Geometry.Polygon(bounding_box)
        
        image_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lte('CLOUD_COVER', 10)) \
            .filter(ee.Filter.eq('IMAGE_QUALITY_OLI', 9))  # only for Landsat 8

        dates = geemap.image_dates(image_collection).getInfo()
        collection_list = image_collection.toList(image_collection.size())
        print("Number of images in this collection: ", collection_list.size().getInfo())
        
        metadata_list_l8 = []

        for i,date in enumerate(dates):
            image = ee.Image(collection_list.get(i))
            metadata=image.getInfo()
            metadata_list_l8.append(metadata)
            message = f"GLIMSID:{glims_id}; CRS:{metadata['bands'][0]['crs']}; CLOUD_COVER:{metadata['properties']['CLOUD_COVER']}; CLOUD_COVER_LAND:{metadata['properties']['CLOUD_COVER_LAND']}; SCENCE_CENTER_TIME:{metadata['properties']['SCENE_CENTER_TIME']}; IMAGE_QUALITY_OLI:{metadata['properties']['IMAGE_QUALITY_OLI']}; IMAGE_QUALITY_TIRS:{metadata['properties']['IMAGE_QUALITY_TIRS']}"
            logging.info(message)
            geemap.ee_export_image(image,
                                filename = os.path.join(out_dir,f"{glims_id}_{date}_L8_C02_T1_L2_SR.tif"),
                                scale = 30,
                                region = region,
                                file_per_band = False)
        try:
            os.mkdir(os.path.join(out_dir, "meta_data"))
        except:
            pass

        pd.Series(metadata_list_l8).to_csv(os.path.join(out_dir,"meta_data", "metadata_list_l8"))

    def export_landsat_seven_images(self, glims_id,bounding_box, start_date,end_date, out_dir):
        
        region = ee.Geometry.Polygon(bounding_box)
        
        image_collection = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lte('CLOUD_COVER', 10)) \
            .filter(ee.Filter.eq('IMAGE_QUALITY', 9))  # only for Landsat 5 and Landsat 7
    
        dates = geemap.image_dates(image_collection).getInfo()

        collection_list = image_collection.toList(image_collection.size())
        print("Number of images in this collection: ", collection_list.size().getInfo())
    
        metadata_list_l7 = []

        for i,date in enumerate(dates):
            image = ee.Image(collection_list.get(i))
            metadata=image.getInfo()
            metadata_list_l7.append(metadata)
            message = f"GLIMSID:{glims_id}; CRS:{metadata['bands'][0]['crs']}; CLOUD_COVER:{metadata['properties']['CLOUD_COVER']}; CLOUD_COVER_LAND:{metadata['properties']['CLOUD_COVER_LAND']}; SCENCE_CENTER_TIME:{metadata['properties']['SCENE_CENTER_TIME']}; IMAGE_QUALITY:{metadata['properties']['IMAGE_QUALITY']};"
            logging.info(message)
            geemap.ee_export_image(image,
                                filename = os.path.join(out_dir,f"{glims_id}_{date}_L7_C02_T1_L2_SR.tif"),
                                scale = 30,
                                region = region,
                                file_per_band = False)
        try:
            os.mkdir(os.path.join(out_dir, "meta_data"))
        except:
            pass
        pd.Series(metadata_list_l7).to_csv(os.path.join(out_dir,"meta_data", "metadata_list_l7"))

    def export_landsat_five_images(self, glims_id,bounding_box, start_date,end_date, out_dir):
    
        region = ee.Geometry.Polygon(bounding_box)

        image_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
            .filterBounds(region) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lte('CLOUD_COVER', 10)) \
            .filter(ee.Filter.eq('IMAGE_QUALITY', 9))  # only for Landsat 5 and Landsat 7
    
        dates = geemap.image_dates(image_collection).getInfo()

        collection_list = image_collection.toList(image_collection.size())
        print("Number of images in this collection: ", collection_list.size().getInfo())
    
        metadata_list_l5 = []

        for i,date in enumerate(dates):
            image = ee.Image(collection_list.get(i))
            metadata = image.getInfo()

            metadata_list_l5.append(metadata)    
            message = f"GLIMSID:{glims_id}; CRS:{metadata['bands'][0]['crs']}; CLOUD_COVER:{metadata['properties']['CLOUD_COVER']}; CLOUD_COVER_LAND:{metadata['properties']['CLOUD_COVER_LAND']}; SCENCE_CENTER_TIME:{metadata['properties']['SCENE_CENTER_TIME']}; IMAGE_QUALITY:{metadata['properties']['IMAGE_QUALITY']};"
            logging.info(message)
            geemap.ee_export_image(image,
                                    filename = os.path.join(out_dir,f"{glims_id}_{date}_L5_C02_T1_L2_SR.tif"),
                                    scale = 30,
                                    region = region,
                                    file_per_band = False)
        try:
            os.mkdir(os.path.join(out_dir, "meta_data"))
        except:
            pass
        pd.Series(metadata_list_l5).to_csv(os.path.join(out_dir,"meta_data", "metadata_list_l5"))

    def export_nasa_dems(self, glims_id, bounding_box, out_dir):
        
        region = ee.Geometry.Polygon(bounding_box)
        image = ee.Image("NASA/NASADEM_HGT/001")
        image = image.clip(region)
        
        geemap.ee_export_image(image,
                            filename = os.path.join(out_dir, f"{glims_id}_NASADEM.tif"),
                            scale = 30,
                            region = region)

    