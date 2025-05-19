import os
import shutil
import tarfile
from scipy import io

TRAIN_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/root/autodl-tmp/imagenet/train'
VAL_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar'
VAL_DEST_DIR = '/root/autodl-tmp/imagenet/val'
VAL_SRC_help_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz'
VAL_DEST_help_DIR = '/root/autodl-tmp/imagenet/'
VAL_DEST_help_DIR1 = '/root/autodl-tmp/imagenet/ILSVRC2012_devkit_t12'


def extract_train():
    with open(TRAIN_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        for i, item in enumerate(tar):
            cls_name = item.name.strip(".tar")
            a = tar.extractfile(item)
            b = tarfile.open(fileobj=a, mode="r:")
            e_path = "{}/{}/".format(TRAIN_DEST_DIR, cls_name)
            if not os.path.isdir(e_path):
                os.makedirs(e_path)
            print("#", i, "extract train dateset to >>>", e_path)
            b.extractall(e_path)


def move_valimg(val_dir=VAL_DEST_DIR, devkit_dir=VAL_DEST_help_DIR1):
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))


def extract_val():
    os.makedirs(VAL_DEST_DIR, exist_ok=True)
    with tarfile.open(VAL_SRC_DIR, 'r:') as tar:
        tar.extractall(path=VAL_DEST_DIR)
    os.makedirs(VAL_DEST_help_DIR, exist_ok=True)
    with tarfile.open(VAL_SRC_help_DIR, 'r:gz') as tar:
        tar.extractall(path=VAL_DEST_help_DIR)
    move_valimg()


if __name__ == '__main__':
    extract_train()
    extract_val()
