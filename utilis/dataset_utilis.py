from torchvision import datasets, transforms
import os
import shutil
import json
from timm.data import create_transform

DEFAULT_CROP_PCT = 0.875
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (.0167 * 255)] * 3)

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    return dataset





def build_transform(is_train, args):
    if args.normalize_from_IMN:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        # print(f'mean:{mean}, std:{std}')
    else:
        if not os.path.exists(os.path.join(args.output_dir, "/pretrain_ds_mean_std.txt")) and not args.eval:
            shutil.copyfile(os.path.dirname(args.finetune) + '/pretrain_ds_mean_std.txt',
                            os.path.join(args.output_dir) + '/pretrain_ds_mean_std.txt')
        with open(os.path.join(os.path.dirname(args.resume)) + '/pretrain_ds_mean_std.txt' if args.eval
                  else os.path.join(args.output_dir) + '/pretrain_ds_mean_std.txt', 'r') as file:
            ds_stat = json.loads(file.readline())
            mean = ds_stat['mean']
            std = ds_stat['std']
            # print(f'mean:{mean}, std:{std}')

    if args.apply_simple_augment:
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=transforms.InterpolationMode.BICUBIC,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            return transform

        # no augment / eval transform
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)  # 256
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))  # 224

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

    else:
        t = []
        if args.input_size < 224:
            crop_pct = input_size / 224
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)  # size = 224
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            # to maintain same ratio w.r.t. 224 images
        )
        # t.append(
        #     transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        #     # to maintain same ratio w.r.t. 224 images
        # )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)
