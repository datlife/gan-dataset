import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.virat_utils import check_stationary_objects, zoom_in_object, generate_object_trajectory, Color


OUTPUT_DIR   = '/media/dat/dataset/VIRAT/outputs'
DEFAULT_PATH = '/media/dat/dataset/VIRAT'

# VIRAT DATA FORMAT

object_anno_fields     = ['object_id', 'object_duration', 'current_frame',
                          'left_top_x', 'left_top_y', 'width', 'height',
                          'object_type']
OBJECT_TYPES           = ['_', 'person', 'car', 'vehicles', 'object', 'bike']
SELECTED_OBJECT_TYPE   = [2, 3]  # only select car or vehicles

SEQUENCE_LEN = 200
OFFSET       = 150

# Naming Convention for the image output file
OUTPUT_FORMAT = '{0}_{1}_{2}.png'


def _main_():
    VIRAT_ANNOT_DIR = os.path.join(DEFAULT_PATH, 'annotations')
    VIRAT_VIDEO_DIR = os.path.join(DEFAULT_PATH, 'videos_original')

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    video_files = os.listdir(VIRAT_VIDEO_DIR)
    video_files = [os.path.splitext(x)[0] for x in video_files]

    print("Found %s videos in current directory" % len(video_files))
    s = time.time()
    for video_file in video_files[0:1]:
        _process_video(filename     = video_file,
                       output_dir   = OUTPUT_DIR,
                       video_source = os.path.join(VIRAT_VIDEO_DIR, video_file + '.mp4'),
                       video_labels = os.path.join(VIRAT_ANNOT_DIR, video_file + '.viratdata.objects.txt'),
                       sequence_len = SEQUENCE_LEN,
                       offset_area  = OFFSET)

    print("Completed in {}", time.time() - s)


def _process_video(filename, output_dir, video_source, video_labels,  sequence_len, offset_area):
    """Load video and generate dataset

    Args:
        filename:
        output_dir:
        video_source:
        video_labels:

    Returns:

    """

    # Load Video file
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise IOError("Please check video path %s" % video_source)

    # Load annotation file
    if not os.path.isfile(video_labels):
        # raise IOError("Cannot file annotation file for %s. Skipping" % anno_path)
        return

    df = pd.read_csv(video_labels, delim_whitespace=True, names=object_anno_fields)
    df = df[df['object_type'].isin(SELECTED_OBJECT_TYPE)]  # Filter object that is not car

    # MAIN PIPELINE
    grouped   = df.groupby('object_id')
    video_dir = os.path.join(output_dir, filename)

    if not os.path.isdir(video_dir):
        os.mkdir(video_dir)

    for idx, group_id in enumerate(grouped.groups):
        object_ground_truths = grouped.get_group(group_id)
        _process_object(cap,
                        ground_truths= object_ground_truths,
                        offset_area  = offset_area,
                        sequence_len = sequence_len,
                        filename     = 'obj_id_' + str(group_id),
                        parent_dir   = video_dir)

    cap.release()


def _process_object(cap, ground_truths, offset_area, sequence_len, filename, parent_dir):
    """

    Args:
        cap:
        ground_truths:
        filename:
        parent_dir:

    Returns:

    """

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    bboxes = ground_truths.loc[:, 'left_top_x': 'height'].values
    bboxes = np.asarray([[(x, y), (x + w, y + h)] for x, y, w, h in bboxes])
    if len(bboxes) == 1:
        return

    # Filter out stationary frames ( e.g: car is parking)
    cut_off_idx = check_stationary_objects(bboxes, duration=sequence_len)
    if cut_off_idx:
        bboxes = bboxes[:cut_off_idx]
    else:
        return  # print("Object in this group is stationary. Skip")

    # ####################
    # Find the first frame
    # ####################

    appear_frames  = ground_truths['current_frame'].values
    starting_frame = np.min(appear_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
    print("\nStaring frame {} ".format(starting_frame))

    num_seqs = int(len(bboxes) / sequence_len)
    bbox_seqs = np.array_split(bboxes, num_seqs)

    # Generate a image mask, size of the clip's dimension
    _, frame = cap.read()
    image_shape = frame.shape

    # ############################
    # Process Each Object in video
    # ############################
    for i, bbox_seq in enumerate(tqdm(bbox_seqs)):
        output_dir = os.path.join(parent_dir, filename + "_seq_" + str(i))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Find crop pts (upper left, lower right)
        upper_left, bottom_right = find_crop_pts(image_shape, bbox_seq, offset_area)

        for idx, (p1, p2) in enumerate(bbox_seq):

            # read and crop image
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

            # Format output filename
            img_path = OUTPUT_FORMAT.format(parent_dir.split('/')[-1], i, idx)

            # save frame to disk
            saved_path = os.path.join(output_dir, img_path)
            cv2.imwrite(saved_path, frame)

        curr_cursor = starting_frame + i * sequence_len
        ground_truths_file = ground_truths.iloc[curr_cursor:curr_cursor + len(bbox_seq)]
        # @TODO: adjust ground truth boxes to crop area

        ground_truths_file.to_csv(os.path.join(output_dir, 'labels.csv'))


def find_crop_pts(image_shape, boxes_list, offset):
    """

    Args:
        image_shape:
        boxes_list:
        offset:

    Returns:

    """

    bboxes = np.array(boxes_list)
    height, width, _ = image_shape

    min_x = max(min(bboxes[:, 0, 0] - offset), 0.)
    min_y = max(min(bboxes[:, 0, 1] - offset), 0.)
    max_x = min(max(bboxes[:, 1, 0] + offset), width)
    max_y = min(max(bboxes[:, 1, 1] + offset), height)
    upper_left = (int(min_x), int(min_y))
    bottom_right = (int(max_x), int(max_y))

    return upper_left, bottom_right

# object_trajectory = generate_object_trajectory(empty_mask, bboxes,
#                                                color=Color.green,
#                                                opacity=60)

# frame = cv2.rectangle(frame, tuple(p1), tuple(p2), color=Color.green, thickness=4)
# frame = cv2.resize(frame, (480, 320), cv2.INTER_LINEAR)

if __name__ == "__main__":
    _main_()

