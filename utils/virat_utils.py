import numpy as np
import cv2


def cut_off_frame(bboxes, duration=50):
    """
    Filter out stationary object (object no longer moving)

    Ags:
      bboxes   - a numpy array shape (num_bboxes, 2, 2)
      duration - an int

    Return:
        First Frame that object not moving after that

    """
    rate_of_change = np.gradient(np.array(bboxes), axis=0)

    # Separate upper bound and lower bound rate of change
    upper_grad = np.sum(np.abs(rate_of_change[:, 0]), axis=1)
    lower_grad = np.sum(np.abs(rate_of_change[:, 1]), axis=1)

    # Calculate rate of change at every duration
    upper_rate = np.add.reduceat(upper_grad, np.arange(0, len(upper_grad), duration))
    lower_rate = np.add.reduceat(lower_grad, np.arange(0, len(lower_grad), duration))

    bbox_rate = upper_rate + lower_rate
    for idx, rate in enumerate(bbox_rate):
        if rate == 0:
            break

    bbox_rate = bbox_rate[:idx]
    return len(bbox_rate) * duration


def generate_focused_area_mask(image, boxes_list, offset, color, thickness=5):
    '''
    Gernate crop area
    '''

    bboxes = np.array(boxes_list)
    height, width, _ = image.shape

    min_x = max(min(bboxes[:, 0, 0] - offset), 0.)
    min_y = max(min(bboxes[:, 0, 1] - offset), 0.)
    max_x = min(max(bboxes[:, 1, 0] + offset), width)
    max_y = min(max(bboxes[:, 1, 1] + offset), height)
    p1 = (int(min_x), int(min_y))
    p2 = (int(max_x), int(max_y))
    mask = np.zeros_like(image)
    mask = cv2.rectangle(mask,
                         p1,
                         p2,
                         color, thickness=5)

    return mask, p1, p2


def generate_object_trajectory(image, boxes_list, color, opacity=50):
    '''
    Visualize object path on video using list of boxes from ground truths
    '''
    copied_img = np.copy(image)
    transparency = opacity / 100.
    for (p1, p2) in boxes_list:
        # Calculate centroid
        x = p1[0] + (p2[0] - p1[0]) / 2.
        y = p1[1] + (p2[1] - p1[1]) / 2.
        centroid = (int(x), int(y))
        drawed_img = cv2.circle(copied_img, centroid, radius=3, color=color)

        # apply the overlay
        cv2.addWeighted(copied_img, 1 - transparency,
                        drawed_img, transparency, 0, copied_img)
    return copied_img


class Color(object):
    '''Simple Color Mapper'''
    red = [255, 0, 0]
    green = [0, 255, 0]
    blue = [0, 0, 255]
    yellow = [255, 255, 0]

    def random(self):
        return np.random.randint(0, 255, size=3)


def filter_ouliner(data, m=2):
    data = data[abs(data - np.mean(data)) < m * np.std(data)]
    if len(data) > 0:
        return data
    else:
        return [np.mean(data)]
