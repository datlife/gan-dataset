import pandas as pd


def get_video_ids(csv_path=None, category='car', min_area=0.1, max_area=0.3, threshold=0.6):
    """
    Return a list of video ids that satisfies the condition as:

       * Contain at least one object of the specified category
       * All the object's area compared to the video size should be:
                * Less than the max_area and greater than min_area and
                * Percentage of satisfied area > threshold

    # This only works for Youtube BBoxes
    # https://research.google.com/youtube-bb/download.html
    """
    if csv_path is None:
        raise IOError("Please specify path to Youtube bounding box file")

    df = pd.read_csv(csv_path, names=['youtube_id', 'timestamp', 'class_id', 'class_name',
                                      'object_id', 'object_presense',
                                      'xmin', 'xmax', 'ymin', 'ymax'])
    df = df[df['class_name'] == category]

    grouped = df.groupby('youtube_id')

    # Only get videos that satisfies min and max area
    grouped = grouped.filter(lambda x: filter_obj(x,
                                                  min_area=min_area,
                                                  max_area=max_area,
                                                  threshold=threshold)).groupby('youtube_id')

    return grouped


def filter_obj(group, min_area=0.05, max_area=0.30, threshold=0.6):

    area = (group['xmax'] - group['xmin']) * (group['ymax'] - group['ymin'])
    area = (area > min_area) & (area < max_area)

    if area.any() is False:
        return False
    else:
        small_areas = area[area == True]
        is_good = (len(small_areas) / (1.0 * len(area))) >= threshold

        return is_good