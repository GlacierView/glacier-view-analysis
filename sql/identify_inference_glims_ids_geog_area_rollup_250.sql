-- Select 250 glaciers per geographic-area rollup for inference.
--
-- Reconstructed 2026-09-19. The original was run from the Athena console on
-- 2024-09-11 and never saved to a file; the console's query history is not
-- readable with the team IAM role (athena:ListQueryExecutions is denied).
--
-- The reconstruction is verified rather than guessed. The query's output
-- survives twice: as an Athena result at
--   s3://glacier-view-athena-output-us-west-1/Unsaved/2024/09/11/
--     21704e69-c2cd-4856-82a1-88ed19a22d27.csv
-- and in the repo as src/glims/data/inference_samples/geog_area_rollup_250.csv
-- The two are byte-identical. That output pins every parameter below:
--
--   1,024 rows                      -> 250 per rollup, with ties
--   max(geog_size_rank) = 250       -> the rank cutoff
--   rank_score only ever 1          -> the most-recent-image filter is ON
--                                      (unlike inference_data_query.sql,
--                                       where it is commented out)
--   is_low_snow_month only ever 1   -> summer-months filter applied
--   max(cloud_cover) = 4.34         -> cloud_cover < 5, not the 10 in the
--                                      design doc, which describes only the
--                                      Earth Engine download filter
--   column order matches this file  -> the SELECT list below is unchanged
--
-- Per-rollup counts in that output: Asia 250, South America 250,
-- North America 251 (a dense_rank tie at 250), Europe 199, Caucausus 74 --
-- the last two having run out of qualifying glaciers.
--
-- This is identical to identify_inference_glims_ids_geog_area_rollup_50.sql
-- except for the final cutoff. It reads the *training* metadata tables
-- (ee_metadata, image_attributes, glims_18k) because glacier selection
-- happens before the inference imagery exists; the _full_time_series_250
-- tables are the product of downloading what this query chose.
--
-- Run against the `glacier-view` database in us-west-1.

select
    *
from (
    select
        *
        , dense_rank() over(partition by geog_area_rollup order by db_area desc) geog_size_rank
    from (
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
        from (
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
                , glims_18k.db_area
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
            and geog_area != 'Canada' -- this will perfectly exclude glaciers on Ellesmere Island which is too far north for DEMs
    )
    where
        rank_score = 1
        and is_low_snow_month = 1
    )
where
    geog_size_rank <= 250