import os
import cv2
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from write_html import writeHTML
from utils.virat_utils import cut_off_frame, generate_focused_area_mask, generate_object_trajectory, Color


# VIRAT DATA FORMAT
OBJECT_TYPES           = ['_', 'person', 'car', 'vehicles', 'object', 'bike']
SELECTED_OBJECT_TYPE   = [2, 3]  # only select car or vehicles
object_anno_fields     = ['object_id', 'object_duration', 'current_frame',
                          'left_top_x', 'left_top_y', 'width', 'height',
                          'object_type']

# @TODO:
# group each chunk into one line


def _main_():

    # PARAMTERS
    CHUNK        = 50
    OFFSET       = 150
    SKIP_FRAMES  = 5
    OUTPUT_DIR   = '/media/dat/dataset1/VIRAT/outputs'
    DEFAULT_PATH = '/media/dat/dataset1/VIRAT'

    # Naming Convention for the image output file
    OUTPUT_FORMAT = '{0}_{1}_{2}.png'

    DEFAULT_ANNOTATION_DIR = os.path.join(DEFAULT_PATH, 'annotations')
    DEFAULT_VIDEO_DIR = os.path.join(DEFAULT_PATH, 'videos_original')

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    video_files = os.listdir(DEFAULT_VIDEO_DIR)
    video_files = [os.path.splitext(x)[0] for x in video_files]

    print("Found %s videos in current directory" % len(video_files))
    html_data = dict()

    for video in video_files[0:10]:

        video_file = video.split('_')
        video_file = "_".join(video_file[:3])
        video_path = os.path.join(DEFAULT_VIDEO_DIR, video + '.mp4')

        # save data for generating HTML file later
        html_data[video_file] = dict()

        # ###############
        # Load Video file
        # ###############
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Cannot open %s" % video_path)

        # ######################
        # Load annotation file
        # ######################
        anno_path = os.path.join(DEFAULT_ANNOTATION_DIR, video + '.viratdata.objects.txt')
        if not os.path.isfile(anno_path):
            print("Cannot file annotation file for %s. Skipping" % anno_path)
            continue  # iterate to next video

        df = pd.read_csv(anno_path, delim_whitespace=True, names=object_anno_fields)

        # ###############################
        # SKIP FRAMES TO VISUALIZE EASIER
        # ###############################
        # @TODO: may need to remove later
        df = df.iloc[::SKIP_FRAMES, :]

        # ############################################
        # Filter out object that is not car or vehicle
        # ###########################################
        object_bboxes = []
        temp_frames   = []
        for _, obj in df.groupby('object_id'):

            object_type = obj['object_type'].values[0]
            if object_type not in SELECTED_OBJECT_TYPE:
                continue

            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            upper_left_pts   = zip(obj['left_top_x'].values, obj['left_top_y'].values)
            width_height_lst = zip(obj['width'].values, obj['height'].values)
            lower_right_pts  = [(px + w, py + h)
                                for (px, py), (w, h) in zip(upper_left_pts, width_height_lst)]

            # Reformat bboxes to opencv format
            bbox_list = [[(px1, py1), (px2, py2)]
                         for px1, py1, (px2, py2) in zip(obj['left_top_x'].values,
                                                         obj['left_top_y'].values,
                                                         lower_right_pts)]
            if bbox_list:
                object_bboxes.append(bbox_list)
                temp_frames.append(obj)

        print("Number of objects in the video %s" % len(object_bboxes))
        grouped_objects = pd.concat(temp_frames).groupby('object_id')

        CURR_VIDEO_DIR = os.path.join(OUTPUT_DIR, video_file)
        if not os.path.isdir(CURR_VIDEO_DIR):
            os.mkdir(CURR_VIDEO_DIR)

        # ############################
        # Process Each Object in video
        # ############################
        for idx, object_id in enumerate(grouped_objects.groups):

            # ############################
            # Filter out stationary frames
            #    e.g: car is parking
            # ############################
            bboxes = object_bboxes[idx]
            cut_off_idx = cut_off_frame(bboxes, duration=CHUNK)
            if cut_off_idx:
                bbox_arr    = np.asarray(bboxes[:cut_off_idx])
                num_chunk   = int(len(bbox_arr) / CHUNK)
                bbox_chunks = np.array_split(bbox_arr, num_chunk)
            else:
                print("Object in this group is stationary. Skip")
                continue

            html_data[video_file][object_id] = dict()
            SUB_SEQ_DIR = os.path.join(CURR_VIDEO_DIR, 'object_' + str(object_id))
            if not os.path.isdir(SUB_SEQ_DIR):
                os.mkdir(SUB_SEQ_DIR)

            # ############################
            # Find the first frame
            # ############################
            appear_frames = grouped_objects.get_group(object_id)['current_frame'].values
            start_frame   = np.min(appear_frames)

            # Generate a image mask, size of the clip's dimension
            # Visualization only
            _, frame = cap.read()
            empty_mask = np.zeros_like(frame)
            object_trajectory = generate_object_trajectory(empty_mask, bboxes,
                                                           color=Color.green,
                                                           opacity=60)

            start = time.time()
            print("Processing object %s of %s" % (object_id, video_file))

            for i, bbox_chunk in enumerate(tqdm(bbox_chunks)):
                focused_area_mask, top_pts, bot_pts = generate_focused_area_mask(empty_mask,
                                                                                 bbox_chunk,
                                                                                 color=Color.yellow,
                                                                                 offset=OFFSET)

                CURR_SEQUENCE_DIR = os.path.join(SUB_SEQ_DIR, str(i))

                if not os.path.isdir(CURR_SEQUENCE_DIR):
                    os.mkdir(CURR_SEQUENCE_DIR)

                image_paths = []
                overview_paths = []
                for idx, (p1, p2) in enumerate(bbox_chunk):
                    # Format file name
                    filename = OUTPUT_FORMAT.format(video_file, i, idx)
                    saved_path = os.path.join(CURR_SEQUENCE_DIR, filename)

                    # Get current frame from clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i * CHUNK + idx*SKIP_FRAMES)
                    _, frame = cap.read()

                    # convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Draw bounding box of the object on current frame
                    frame = cv2.rectangle(frame, (p1[0], p1[1]), (p2[0], p2[1]), color=Color.green, thickness=4)

                    # Add object flow mask to current frame
                    frame = cv2.addWeighted(frame, 1.0, object_trajectory, 0.2, 0)

                    # Crop area
                    frame = frame[top_pts[1]:bot_pts[1], top_pts[0]:bot_pts[0]]

                    # Resize to save disk space
                    frame = cv2.resize(frame, (480, 320), cv2.INTER_LINEAR)

                    cv2.imwrite(saved_path, frame)

                    # For writing HTMl file
                    image_paths.append(filename)
                    overview_paths.append(saved_path)

                # save annotation file to current sub directory
                labels = grouped_objects.get_group(object_id).iloc[start_frame + i*CHUNK:
                                                                  start_frame + i*CHUNK + len(bbox_chunk)]
                labels.to_csv(os.path.join(CURR_SEQUENCE_DIR, 'labels.csv'))
                writeHTML(filename=os.path.join(CURR_SEQUENCE_DIR, 'result.html'),
                          html_template='subset_gallery.html',
                          image_paths=image_paths)

                # Save all the paths fo
                html_data[video_file][object_id][i] = overview_paths

            print("Done in {}".format(time.time() - start))
            cap.release()

    # Output Overview
    writeHTML(filename=os.path.join(OUTPUT_DIR, 'overview.html'),
              html_template='overview.html',
              videos=html_data)


if __name__ == "__main__":
    _main_()
