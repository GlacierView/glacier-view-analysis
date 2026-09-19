select
    *
from
    (
    select
        *
        , row_number() over(partition by glims_id order by src_date desc) as rank_score
        ,   case
            when 
                is_southen_hemisphere = 0
                and (month_number >= 5  and month_number <= 10)
                then 1
            when
                is_southen_hemisphere = 1
                and (month_number >= 11 or month_number <= 4)
                then 1
            end as is_low_snow_month
    from
        (
        select
            eem.file_name
            , eem.glims_id
            , eem.cloud_cover
            , eem.image_quality
            , eem.image_quality_oli
            , eem.spacecraft_id
            , cast(eem.src_date as timestamp) as src_date
            , month(cast(eem.src_date as timestamp)) as month_number
            , ia.height_in_pixels*ia.width_in_pixels*num_of_bands as num_pixels
            , ia.no_data_pixel_count
            , ia.zero_pixel_count
            , cast(ia.zero_pixel_count as decimal(38,19))/(ia.height_in_pixels*ia.width_in_pixels*num_of_bands) as percentage_zero_pixels
            , ia.negative_pixel_count
            , glims_18k.geog_area_rollup
            , glims_18k.geog_area
            , case
                when glims_18k.geog_area_rollup in ('South America', 'Oceania') then 1
                else 0
              end as is_southen_hemisphere
        from
            ee_metadata eem
            join image_attributes ia on 
                eem.file_name = ia.file_name
            join glims_18k on
                eem.glims_id = glims_18k.glac_id
        )
    where
         cloud_cover < 5
         and (image_quality = 9 or image_quality_oli = 9)
         and num_pixels > 50000
         and no_data_pixel_count = 0
         and percentage_zero_pixels < 0.1
         and geog_area != 'Canada' -- this will perfectly exclude glaciers on ellesmere island which is too far north for DEMs
    )
where
    rank_score = 1
    and is_low_snow_month = 1