import os
import cv2
import time
import numpy as np
import pandas as pd

from tqdm import tqdm
from write_html import writeHTML
from utils.virat_utils import check_stationary_objects, zoom_in_object, generate_object_trajectory, Color


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
    SEQUENCE_LEN = 50
    OFFSET = 200
    SKIP_FRAMES = 7
    OUTPUT_DIR   = '/media/dat/dataset1/VIRAT/outputs'
    DEFAULT_PATH = '/media/dat/dataset1/VIRAT'

    # Naming Convention for the image output file
    OUTPUT_FORMAT = '{0}_{1}_{2}.png'

    VIRAT_ANNOT_DIR = os.path.join(DEFAULT_PATH, 'annotations')
    VIRAT_VIDEO_DIR = os.path.join(DEFAULT_PATH, 'videos_original')

    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    video_files = os.listdir(VIRAT_VIDEO_DIR)
    video_files = [os.path.splitext(x)[0] for x in video_files]

    print("Found %s videos in current directory" % len(video_files))
    html_data = dict()

    for video_file in video_files:

        video_path = os.path.join(VIRAT_VIDEO_DIR, video_file + '.mp4')
        anno_path = os.path.join(VIRAT_ANNOT_DIR, video_file + '.viratdata.objects.txt')

        # ###############
        # Load Video file
        # ###############
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError("Please check video path %s" % video_path)

        # ####################
        # Load annotation file
        # ####################
        if not os.path.isfile(anno_path):
            # raise IOError("Cannot file annotation file for %s. Skipping" % anno_path)
            continue

        df = pd.read_csv(anno_path, delim_whitespace=True, names=object_anno_fields)

        # Filter object that is not in SELECTED_OBJECT_TYPES (Car)
        df = df[df['object_type'].isin(SELECTED_OBJECT_TYPE)]
        df = df.iloc[::SKIP_FRAMES, :]

        # #############
        # MAIN PIPELINE
        # #############
        start = time.time()
        CURR_VIDEO_DIR = os.path.join(OUTPUT_DIR, video_file)
        if not os.path.isdir(CURR_VIDEO_DIR):
            os.mkdir(CURR_VIDEO_DIR)

        grouped = df.groupby('object_id')

        for idx, group_id in enumerate(grouped.groups):
            # ############################
            # Filter out stationary frames
            #    e.g: car is parking
            # ############################
            # Convert (x, y, w, h) to (x1, y1, x2, y2) in openCV format
            bboxes = grouped.get_group(group_id).loc[:, 'left_top_x': 'height'].values
            bboxes = np.asarray([[(x, y), (x + w, y + h)] for x, y, w, h in bboxes])
            if len(bboxes) == 1:
                continue

            cut_off_idx = check_stationary_objects(bboxes, duration=SEQUENCE_LEN)
            if cut_off_idx:
                bboxes = bboxes[:cut_off_idx]
            else:
                # print("Object in this group is stationary. Skip")
                continue

            num_seqs = int(len(bboxes) / SEQUENCE_LEN)
            bbox_seqs = np.array_split(bboxes, num_seqs)
            # ####################
            # Find the first frame
            # ####################
            appear_frames = grouped.get_group(group_id)['current_frame'].values
            start_frame = np.min(appear_frames)

            # Generate a image mask, size of the clip's dimension
            _, frame = cap.read()
            empty_mask = np.zeros_like(frame)
            object_trajectory = generate_object_trajectory(empty_mask, bboxes,
                                                           color=Color.green,
                                                           opacity=60)

            SUB_SEQ_DIR = os.path.join(CURR_VIDEO_DIR, 'object_' + str(group_id))
            if not os.path.isdir(SUB_SEQ_DIR):
                os.mkdir(SUB_SEQ_DIR)

            # ############################
            # Process Each Object in video
            # ############################
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - SKIP_FRAMES * 1)
            for i, bbox_seq in enumerate(tqdm(bbox_seqs)):
                focused_area_mask, top_pts, bot_pts = zoom_in_object(empty_mask,
                                                                     bbox_seq,
                                                                     color=Color.yellow,
                                                                     offset=OFFSET)

                CURR_SEQUENCE_DIR = os.path.join(SUB_SEQ_DIR, str(i))
                if not os.path.isdir(CURR_SEQUENCE_DIR):
                    os.mkdir(CURR_SEQUENCE_DIR)

                for idx, (p1, p2) in enumerate(bbox_seq):
                    # Format file name
                    filename = OUTPUT_FORMAT.format(video_file, i, idx)
                    saved_path = os.path.join(CURR_SEQUENCE_DIR, filename)

                    # Get current frame from clip
                    for _ in range(SKIP_FRAMES):
                        _, frame = cap.read()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Draw bounding box of the object on current frame
                    frame = cv2.rectangle(frame, tuple(p1), tuple(p2), color=Color.green, thickness=4)

                    # Add object flow mask to current frame
                    frame = cv2.addWeighted(frame, 1.0, object_trajectory, 0.2, 0)
                    # Crop area
                    frame = frame[top_pts[1]:bot_pts[1], top_pts[0]:bot_pts[0]]

                    frame = cv2.resize(frame, (480, 320), cv2.INTER_LINEAR)
                    cv2.imwrite(saved_path, frame)

                # save annotation file to current sub directory
                ground_truths_file = grouped.get_group(group_id).iloc[
                         start_frame + i * SEQUENCE_LEN:start_frame + i * SEQUENCE_LEN + len(bbox_seq)]

                ground_truths_file.to_csv(os.path.join(CURR_SEQUENCE_DIR, 'labels.csv'))

        print("Processed video %s in %.2f sec(s)." % (video_file, time.time() - start))
        cap.release()


if __name__ == "__main__":
    _main_()

# # For writing HTMl file
# image_paths.append(filename)
# summary.append(saved_path)
#     writeHTML(filename=os.path.join(CURR_SEQUENCE_DIR, 'result.html'),
#               html_template='subset_gallery.html',
#               image_paths=image_paths)
#
#     # Save summary
#     html_data[video_file][group_id][i] = summary
#
# # Generate new video clip
# if len(preview_video):
#     preview_video_path = os.path.join(CURR_VIDEO_DIR, 'preview.mp4')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     new_clip = ImageSequenceClip(preview_video, fps=fps)
#     new_clip.write_videofile(preview_video_path, fps=fps, codec='mpeg4', bitrate='1e11', verbose=0)
#
# # Output Overview
# writeHTML(filename=os.path.join(OUTPUT_DIR, 'overview.html'),
#           html_template='overview.html',
#           videos=html_data)
