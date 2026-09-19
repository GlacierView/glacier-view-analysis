# GlacierView LLD

## Earth Engine Data Extraction

References:
https://developers.google.com/earth-engine/guides/ic_filtering


### Glacier Inference Selection

The selection for inference is determined by following SQL query [identify_inference_glims_ids_geog_area_rollup_50.sql](https://github.com/mattwaismann/glacier-view-analysis/blob/main/src/sql/identify_inference_glims_ids_geog_area_rollup_50.sql)  which runs over our AWS Athena metadata database. 

This query works by scanning over all of our **training data** metadata for GLIMS IDs that have at least one file with cloud_cover < 10, image_quality = 9, more than 50,000 pixels, 0 no data pixels, less than 10% zero pixels, and have geographic area that’s not Canada (since those glaciers all appear on Ellesmere Island and are too far north for DEMs), and has an image in a low snow month (all of them do). Then we will take 50 GLIMS IDs per our 5 geographic area rollups based on db_area.

In September 2024, we can an effort to expand the since of our inference dataset. We took the existing query and extended it to take 250 glaciers per of the 5 geographic area rollups. This time, when downloading the images from the Earth Engine API, we decided to push certain filters down to the server, so we could limit the amount of glaciers we’re downloading which would only later get filtered out in our metadata filtering step. 

### Image Collection Filtering

At the heart of the Earth Engine is the **`ImageCollection`**, an object which represents a sequence of images. An `ImageCollection` can be loaded by passing an Earth Engine **asset ID** into the `ImageCollection` constructor. You can find a list of all asset IDs in the [Data Catalog](https://developers.google.com/earth-engine/datasets) (relevant datasets are Landsat and Sentinel imagery, digital elevation models (DEMs) for terrain, and there’s also MODIS imagery and NAIP high-res imagery that are worth exploring).

The below `ImageCollection` will contain every Landsat 5 image in the collection 2 (newest), tier 1 (highest data quality), level 2 (surface reflectance, correct for atmospheric effects) in the public library (which is a lot). Therefore, we usually will want to filter the `ImageCollection`.

```
landsat_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
```

Based on our previous deep dives in image metadata and their relationship with image quality for time series glacier segmentation, we will use the following filters. 

```
landsat_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') \
    .filterBounds(region) \ # the bounding box around the glacier
    .filterDate(start_date, end_date) \ # the date interval we want images for
    .filter(ee.Filter.lte('CLOUD_COVER', 10)) \
    .filter(ee.Filter.eq('IMAGE_QUALITY', 9)) \ # only for Landsat 5 and Landsat 7
    .filter(ee.Filter.eq('IMAGE_QUALITY_OLI', 9)) \ # only for Landsat 8 

```

*Risk - we lose out on a dense time series by applying these filters. Therefore, these data may not be as useful if we ever extend beyond a time series glacier retreat meta-analysis.* 

By applying these filters, we expect to reduce the `ImageCollection` size by **63%** (this number came from cloud cover being set to 5, we later changed it to 10) (based on the metadata in 248,250 files of unfiltered images we have pulled from the various Landsat image collections.).

**Could we have pushed the filtered for summer months to the server?**
... yeah probably, next time.



### Uploading images and DEMs to our AWS Cloud storage (S3)

After navigating to the local directory which contains our landsat images, we upload to S3 with the following command
`aws s3 cp . s3://full-time-series-images-t1-l2-sr --recursive`. We found the `sync` command to be too slow.

### Metadata collection

#### Earth Engine provided metadata

We persist the metadata provided by earth engine. Earth engine provides these metadata via the `Image` object’s `getInfo()` method. This call returns a JSON blob of metadata for each of the individual images being downloaded (this is done iteratively, at the same time as download time).  At the same time as we download these images, we persist these JSON blobs in a list per GLIMS ID per Landsat satellite (tech debt - we could’ve been a lot cleaner with how we store this metadata - the script which relationalizes these metadata has to navigate a strange file structure)

*Example folder structure of list of of JSON blobs:* 

* landsat/
    * G007068E45470N
        * G007068E45470N_1984-04-18_L5_C02_T1_L2_SR.tif
        * ...
        * G007068E45470N_2024-01-01_L8_C02_T1_L2_SR.tif
        * meta_data/
            * metadata_list_l5
            * metadata_list_l7
            * metadata_list_l8
    * ... other GLIMS IDs


Note that earth engine may sometimes provide more than one metadata record per image (having two different cloud cover values for the same image is a common scenario that we see), so we perform deduplication, choosing one of the records arbitrarily to ensure we have one record per file name.

#### Self generated metadata

In addition to the metadata provided by Earth Engine, we also generate our own metadata:

**Logs**
 **** As each image is downloaded from earth engine, we collect logs for the purpose of monitoring the download process. Note, these metadata are really only helpful for monitoring download progress because all of the metadata fields provided here come directly from the Earth Engine provided metadata.

_*Example log record:*_

*<image download timestamp>:<log level>: <GLIMS_ID>; <CRS>; <CLOUD_COVER>:<CLOUD_CLOVER_LAND>; <SCENE_CENTER_TIME>;<IMAGE_QUALITY>;*

```
2024-09-11 17:41:50,126:INFO:GLIMSID:G087896E49098N; CRS:EPSG:32645; CLOUD_COVER:5; CLOUD_COVER_LAND:5; SCENCE_CENTER_TIME:04:16:39.3950880Z; IMAGE_QUALITY:9;
```

**Image and file attributes**
After every image has been downloaded. We iterate over each downloaded file, collect metadata about the file itself and open the image and collect metadata about the image itself. 

_*File attributes collected:*_ file_name, src_date, landsat, size_in_bytes

*_Image attributes collected:_* file_name, height_in_pixels, width_in_pixels, left_bound, right_bound, top_bound, bottom_bound, num_of_bands, epsg_code, no_data_val, pixel_width_res, pixel_height_res, zero_pixel_count, no_data_pixel_count, negative_pixel_count


### Metadata processing and loading into our database

The *time_series_metadata_handler.ipynb* file processes (and collects in the case of the image and file attributes) all of our metadata into relational CSV files, where each record belongs to a single file (file name primary key). These CSV files are then uploaded to Amazon S3, registered in AWS Glue as individual data tables for easy consumption with a tool like AWS Athena.

Database name: glacier-view
Table names: `ee_metadata_full_time_series_250`, `file_attributesee_metadata_full_time_series_250`, `image_attributesee_metadata_full_time_series_250`, `processed_logsee_metadata_full_time_series_250`

These files were uploaded to `s3://full-time-series-metadata-t1-l2-sr` and crawled over by the [full-time-series-metadata-t1-l2-sr-crawler](https://us-west-1.console.aws.amazon.com/glue/home?region=us-west-1#/v2/data-catalog/crawlers/view/full-time-series-metadata-t1-l2-sr-crawler) in us-west-1.


## Inference

### Pre-inference filtering

Before applying our U-Net for our semantic segmentation task on our extracted satellite images, we have a metadata-based filtering step where we only run inference for images which meet our criteria for inference. This criteria is applied via a SQL query (Github file link here) and the actual filtering happens before the images read into local memory with rasterio.  

_*Filter criteria:*_

* cloud cover: less than 10% of the image
* summer months only: May-October in the nothern hemisphere and November-April in the southern hemisphere
* number of pixels > 50000
    * we should break down the 1,000,000 ways this is a dumb thing to do
* no data pixel count = 0
* percentage_zero_pixels < 0.1 

### The model

The model is stored in a file called `unet_summer_model_unfrozen_100`. This model expects an input of shape SHAPE and bands presented in the following order:

1. NDSI
2. NDWI
3. Blue
4. Green
5. Red
6. Near infrared (NIR)
7. Shortwave infrared (SWIR)
8. Thermal
9. DEM

### The analysis

The areas_true_vs_predicted notebook needs the following input:

* areas_binary_05_filtered_smooth_25.csv (comes from final_areas.py)
* training_areas.csv (somansh will push to github and email me)
* training_data_set.csv (somansh will push to github and email me)
* geo_areas.csv (somansh will push to github and email me)

### Gotchas

1. We should pick to resize based on a single size otherwise we get two lines appearing in the line plot
2. 

