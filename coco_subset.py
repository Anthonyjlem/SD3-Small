import os
import json


def copy_image_subset(subset_dir, src_dir, ann_file, num_images):
    os.makedirs(subset_dir, exist_ok=True)
    with open(ann_file, "r") as f:
        ann = json.load(f)
    keep_images = {img["file_name"] for img in ann["images"][:num_images]}
    num_files = 0
    for fname in os.listdir(src_dir):
        if fname in keep_images:
            os.system(f"cp {src_dir}/{fname} {subset_dir}/")
            num_files += 1
            if num_files == num_images:
                break


def filter_img_entries(data, kept_files):
    kept_images = []
    kept_image_ids = set()
    for img in data["images"]:
        if img["file_name"] in kept_files:
            kept_images.append(img)
            kept_image_ids.add(img["id"])
    return kept_images, kept_image_ids


def make_ann_subset(subset_dir, org_json, subset_json):
    kept_files = set(os.listdir(subset_dir))
    with open(org_json, "r") as f:
        data = json.load(f)
    kept_images, kept_image_ids = filter_img_entries(data, kept_files)
    kept_annotations = [
        ann for ann in data["annotations"]
        if ann["image_id"] in kept_image_ids
    ]
    filtered = {
        "images": kept_images,
        "annotations": kept_annotations,
        "licenses": data.get("licenses", []),
        "info": data.get("info", []),
    }
    os.makedirs(os.path.dirname(subset_json), exist_ok=True)
    with open(subset_json, "w") as f:
        json.dump(filtered, f)


if __name__ == "__main__":
    with open("../Datasets/COCO_subset/annotations_trainval2017/captions_train2017.json", "r") as f:
        ann = json.load(f)
    for i in range(10000):
        if ann["annotations"][i]["image_id"] == 40881:
            print(ann["annotations"][i])
    # copy_image_subset("../Datasets/COCO_subset/train2017",
    #             "../Datasets/COCO/train2017",
    #             "../Datasets/COCO/annotations_trainval2017/captions_train2017.json",
    #             640)
    # copy_image_subset("../Datasets/COCO_subset/val2017",
    #             "../Datasets/COCO/val2017",
    #             "../Datasets/COCO/annotations_trainval2017/captions_val2017.json",
    #             272)
    # make_ann_subset("../Datasets/COCO_subset/train2017",
    #                 "../Datasets/COCO/annotations_trainval2017/captions_train2017.json",
    #                 "../Datasets/COCO_subset/annotations_trainval2017/captions_train2017.json")
    # make_ann_subset("../Datasets/COCO_subset/val2017",
    #                 "../Datasets/COCO/annotations_trainval2017/captions_val2017.json",
    #                 "../Datasets/COCO_subset/annotations_trainval2017/captions_val2017.json")