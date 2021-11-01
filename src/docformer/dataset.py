# -*- coding: utf-8 -*-
import os
import pickle

import pytesseract

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor


def normalize_box(box, width, height):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.
    """
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def resize_align_bbox(bbox, original_image, target_size):
    x_, y_ = original_image.size
    x_scale = target_size / x_
    y_scale = target_size / y_
    orig_left, orig_top, orig_right, orig_bottom = tuple(bbox)
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    xmax = int(np.round(orig_right * x_scale))
    ymax = int(np.round(orig_bottom * y_scale))
    return [x, y, xmax, ymax]


def apply_ocr(example):
    """
    Returns words and its bounding boxes from an image
    """
    image = Image.open(example)
    width, height = image.size

    ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text)
    words = [str(w) for w in words]
    coordinates = ocr_df[["left", "top", "width", "height"]]
    actual_bboxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_bbox = [
            x,
            y,
            x + w,
            y + h,
        ]  # we turn it into (left, top, left+width, top+height) to get the actual box
        actual_bboxes.append(actual_bbox)

    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}


def get_tokens_with_bboxes(tokenizer,
                           words,
                           unnormalized_word_boxes,
                           normalized_word_boxes):
    token_boxes = []
    final_word_tokens = []
    unnormalized_token_boxes = []
    trio = zip(words, unnormalized_word_boxes, normalized_word_boxes)
    for word, unnormalized_box, box in trio:
        word_tokens = tokenizer.tokenize(word)
        final_word_tokens.extend(word_tokens)
        unnormalized_token_boxes.extend(
            unnormalized_box for _ in range(len(word_tokens))
        )
        token_boxes.extend(box for _ in range(len(word_tokens)))

    return (
        token_boxes,
        unnormalized_token_boxes,
        final_word_tokens,
    )


def get_centroid(actual_bbox):
    centroid = []
    for i in actual_bbox:
        width = i[2] - i[0]
        height = i[3] - i[1]
        centroid.append([i[0] + width / 2, i[1] + height / 2])
    return centroid


def get_relative_distance(actual_bbox, centroid, index):
    """
    Note: We take absolute distances to avoid embedding problems.
    Reason is that all we want to know is how FAR one word is
    from the other (in x-dim). A change of line will be reflected
    in the y-dim.
    """
    a_rel_x = []
    a_rel_y = []

    for i in range(0, len(actual_bbox) - 1):
        if i != index:
            prev = actual_bbox[i]
            curr = actual_bbox[i + 1]
            a_rel_x.append(
                [
                    prev[0],  # top left x
                    prev[2],  # bottom right x
                    abs(prev[2] - prev[0]),  # width
                    abs(curr[0] - prev[0]),  # diff top left x
                    abs(curr[0] - prev[0]),  # diff bottom left x
                    abs(curr[2] - prev[2]),  # diff top right x
                    abs(curr[2] - prev[2]),  # diff bottom right x
                    abs(centroid[i + 1][0] - centroid[i][0]),
                ]
            )
            a_rel_y.append(
                [
                    prev[1],  # top left y
                    prev[3],  # bottom right y
                    abs(prev[3] - prev[1]),  # height
                    abs(curr[1] - prev[1]),  # diff top left y
                    abs(curr[3] - prev[3]),  # diff bottom left y
                    abs(curr[1] - prev[1]),  # diff top right y
                    abs(curr[3] - prev[3]),  # diff bottom right y
                    abs(centroid[i + 1][1] - centroid[i][1]),
                ]
            )
        else:
            a_rel_x.append([0] * 8)  # For the actual last word
            a_rel_y.append([0] * 8)  # For the actual last word

    a_rel_x.append([0] * 8)  # For the last word
    a_rel_y.append([0] * 8)  # For the last word
    return a_rel_x, a_rel_y


def apply_mask(inputs):
    inputs = torch.as_tensor(inputs)
    rand = torch.rand(inputs.shape)
    # where the random array is less than 0.15, we set true
    mask_arr = (rand < 0.15) * (inputs != 101) * (inputs != 102) * (inputs != 0)
    # create selection from mask_arr
    selection = torch.flatten(mask_arr.nonzero()).tolist()
    # apply selection index to inputs.input_ids, adding MASK tokens
    inputs[selection] = 103
    return inputs


def read_image_and_extract_text(image):
    original_image = Image.open(image).convert("RGB")
    return apply_ocr(image)


def create_features(
        image,
        path_to_save,
        tokenizer,
        target_size=224,
        max_seq_length=512,
        save_to_disk=False,
):

    pad_token_box = [0, 0, 0, 0]
    original_image = Image.open(image).convert("RGB")
    entries = apply_ocr(image)
    resized_image = original_image.resize((target_size, target_size))
    unnormalized_word_boxes = entries["bbox"]
    words = entries["words"]

    width, height = original_image.size
    normalized_word_boxes = [
        normalize_box(bbox, width, height) for bbox in unnormalized_word_boxes
    ]
    assert len(words) == len(normalized_word_boxes)

    token_boxes, unnormalized_token_boxes, final_word_tokens = get_tokens_with_bboxes(
        tokenizer, words, unnormalized_word_boxes, normalized_word_boxes
    )

    # Truncation of token_boxes + token_labels
    special_tokens_count = 1
    remaining_length = max_seq_length - special_tokens_count
    if len(token_boxes) > remaining_length:
        token_boxes = token_boxes[: remaining_length]
        unnormalized_token_boxes = unnormalized_token_boxes[: remaining_length]

    # add bounding boxes and labels of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes
    unnormalized_token_boxes = ([[0, 0, 0, 0]] + unnormalized_token_boxes)
    encoding = tokenizer(" ".join(words),
                         padding="max_length",
                         max_length=max_seq_length,
                         truncation=True,
                         add_special_tokens=False)
    # add CLS token manually to avoid autom. addition of SEP
    encoding["input_ids"] = [101] + encoding["input_ids"][:-1]

    # Padding of token_boxes up the bounding boxes to the sequence length.
    input_ids = tokenizer(" ".join(words), truncation=True)["input_ids"]
    assert len(input_ids) == len(token_boxes)  # check if number of tokens match
    padding_length = max_seq_length - len(input_ids)
    token_boxes += [pad_token_box] * padding_length
    unnormalized_token_boxes += [pad_token_box] * padding_length
    encoding["bbox"] = token_boxes
    encoding["mlm_labels"] = encoding["input_ids"]

    assert len(encoding["mlm_labels"]) == max_seq_length
    assert len(encoding["input_ids"]) == max_seq_length
    assert len(encoding["attention_mask"]) == max_seq_length
    assert len(encoding["token_type_ids"]) == max_seq_length
    assert len(encoding["bbox"]) == max_seq_length

    encoding["resized_image"] = ToTensor()(resized_image)

    # Applying mask for the sake of pre-training
    encoding["input_ids"] = apply_mask(encoding["input_ids"])

    # rescale and align the bounding boxes to match the resized image size (typically 224x224)
    new_bboxes = []
    for bbox in unnormalized_token_boxes:
        new_bboxes.append(resize_align_bbox(bbox, original_image, target_size))
    
    encoding["resized_and_aligned_bounding_boxes"] = new_bboxes
    index = -1
    
    # getting the index of the last word (using the input id)
    for i in range(len(encoding["input_ids"]) - 1):
        if encoding["input_ids"][i + 1] == 0:
            index = i
            break

    # adding the relative position as well
    actual_bbox = encoding["resized_and_aligned_bounding_boxes"]
    centroid = get_centroid(actual_bbox)

    a_rel_x, a_rel_y = get_relative_distance(actual_bbox, centroid, index)
    encoding["unnormalized_token_boxes"] = unnormalized_token_boxes

    for k, v in encoding.items():
        encoding[k] = torch.as_tensor(encoding[k])

    encoding.update({
        "x_features": torch.as_tensor(a_rel_x),
        "y_features": torch.as_tensor(a_rel_y),
        })
    assert torch.lt(encoding["x_features"], 0).sum().item() == 0
    assert torch.lt(encoding["y_features"], 0).sum().item() == 0
    if save_to_disk:
        os.makedirs(path_to_save, exist_ok=True)
        image_name = os.path.basename(image)
        with open(f"{path_to_save}{image_name}.pickle", "wb") as f:
            pickle.dump(encoding, f)
    return encoding
