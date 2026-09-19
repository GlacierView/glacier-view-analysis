select
    eem.*
    , ia.height_in_pixels*ia.width_in_pixels*ia.num_of_bands as num_pixels
    , ia.no_data_pixel_count
    , ia.zero_pixel_count
    , cast(ia.zero_pixel_count as decimal(38,19))/cast(ia.height_in_pixels*ia.width_in_pixels*ia.num_of_bands as decimal(38,19)) as percentage_zero_pixels
    , ia.negative_pixel_count
from
    (
    select
        *
    from
        (
        select
            file_name
            , glims_id
            , cloud_cover
            , coalesce(image_quality, image_quality_oli) as image_quality
            , row_number() over(partition by file_name order by "version" desc) as rank
        from
            ee_metadata
        )
    where
        rank = 1
    ) eem
    join image_attributes ia on
        eem.file_name = ia.file_name

