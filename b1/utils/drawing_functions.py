import cv2
import numpy as np


def draw_rectangle_with_text(img, pt1, pt2, color, thickness=2,
                             text=None, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8,
                             text_color=(255, 255, 255), text_thickness=2):

    cv2.rectangle(img, pt1, pt2, color, thickness)

    if text:
        draw_multiline_text(img, text, pt1, font_face, font_scale, text_color,
                            text_thickness, bg_color=color)


def draw_multiline_text(img, text, org, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=0.8, font_color=(255, 255, 255),
                        thickness=2, bg_color=(0, 165, 255)):
    if '\n' in text:
        uv_top_left = np.array([org[0], org[1]+10], dtype=float)
        for line in text.split('\n'):
            (text_width, text_height) = cv2.getTextSize(
                line, font_face, fontScale=font_scale, thickness=thickness)[0]
            uv_bottom_left_i = uv_top_left + [0, text_height]
            org = tuple(uv_bottom_left_i.astype(int))

            box_coords = (org, (org[0] + text_width + 2, org[1] - text_height - 6))
            cv2.rectangle(img, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
            cv2.putText(img, line, org, font_face, font_scale, font_color, thickness)
            uv_top_left += [0, text_height * 1.5]
    else:
        (text_width, text_height) = cv2.getTextSize(
            text, font_face, fontScale=font_scale, thickness=thickness)[0]
        cv2.rectangle(img, org, (org[0] + text_width + 2, org[1] +
                                 text_height + 10), bg_color, cv2.FILLED)
        cv2.putText(img, text, (org[0], org[1] + text_height + 3),
                    font_face, 0.8, font_color, thickness)
