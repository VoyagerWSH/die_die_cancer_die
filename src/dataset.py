import lightning.pytorch as pl
import torchvision
import medmnist
import torch
import torchio as tio
import math
import numpy as np
import json
import tqdm
import os
from collections import Counter
import pickle

class PathMnist(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for PathMnist dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, use_data_augmentation=True, batch_size=1024, num_workers=8, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.prepare_data_transforms()

    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        if self.use_data_augmentation:
            self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomRotation(degrees=(-60, 60)),
                torchvision.transforms.GaussianBlur(kernel_size=3)
            ])
        else:
            self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

    def prepare_data(self):
        medmnist.PathMNIST(root='../data', split='train', download=True, transform=self.train_transform)
        medmnist.PathMNIST(root='../data', split='val', download=True, transform=self.test_transform)
        medmnist.PathMNIST(root='../data', split='test', download=True, transform=self.test_transform)

    def setup(self, stage=None):
        self.train = medmnist.PathMNIST(root='../data', split='train', download=True, transform=self.train_transform)
        self.val = medmnist.PathMNIST(root='../data', split='val', download=True, transform=self.test_transform)
        self.test = medmnist.PathMNIST(root='../data', split='test', download=True, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, persistent_workers=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

# Voxel spacing is space between pixels in orig 512x512xN volumes
# "pixel_spacing" stored in sample dicts is also in orig 512x512xN volumes
VOXEL_SPACING = (0.703125, 0.703125, 2.5)
CACHE_IMG_SIZE = [256, 256]
class NLST(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for NLST dataset. This will load the dataset, as used in https://ascopubs.org/doi/full/10.1200/JCO.22.01345.

        The dataset has been preprocessed for you fit on each CPH-App nodes NVMe SSD drives for faster experiments.
    """



    def __init__(
            self,
            use_data_augmentation=False,
            batch_size=1,
            num_workers=0,
            nlst_metadata_path="/wynton/protected/project/cph/cornerstone/nlst-metadata/full_nlst_google.json",
            valid_exam_path="/wynton/protected/project/cph/cornerstone/nlst-metadata/valid_exams.p",
            nlst_dir="/scratch/datasets/nlst/preprocessed",
            lungrads_path="/wynton/protected/project/cph/cornerstone/nlst-metadata/nlst_acc2lungrads.p",
            num_images=200,
            max_followup=6,
            img_size = [256, 256],
            class_balance=False,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_followup = max_followup

        self.nlst_metadata_path = nlst_metadata_path
        self.nlst_dir = nlst_dir
        self.num_images = num_images
        self.img_size = img_size
        self.valid_exam_path = valid_exam_path
        self.class_balance = class_balance
        self.lungrads_path = lungrads_path

        self.prepare_data_transforms()

    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''
        resample = tio.transforms.Resample(target=VOXEL_SPACING)
        padding = tio.transforms.CropOrPad(
            target_shape=tuple(CACHE_IMG_SIZE + [self.num_images]), padding_mode=0
        )
        resize = tio.transforms.Resize(self.img_size + [self.num_images])


        self.train_transform = tio.transforms.Compose([
            resample,
            padding,
            resize
        ])

        if self.use_data_augmentation:
            # TODO: Support some data augmentations. Hint: consider using torchio.
            raise NotImplementedError("Not implemented yet")


        self.test_transform = tio.transforms.Compose([
            resample,
            padding,
            resize
        ])

        self.normalize = torchvision.transforms.Normalize(mean=[128.1722], std=[87.1849])

    def setup(self, stage=None):
        self.metadata = json.load(open(self.nlst_metadata_path, "r"))
        self.acc2lungrads = pickle.load(open(self.lungrads_path, "rb"))
        self.valid_exams = set(torch.load(self.valid_exam_path))
        self.train, self.val, self.test = [], [], []

        for mrn_row in tqdm.tqdm(self.metadata, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            dataset = {"train": self.train, "dev": self.val, "test": self.test}[split]

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():

                    exam_str = "{}_{}".format(exam_dict["exam"], series_id)

                    if exam_str not in self.valid_exams:
                        continue


                    exam_int = int(
                        "{}{}{}".format(int(pid), int(exam_dict["screen_timepoint"]), int(series_id.split(".")[-1][-3:]))
                    )

                    y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, exam_dict["screen_timepoint"])
                    sample = {
                        "pid": pid,
                        "exam_str": exam_str,
                        "exam_int": exam_int,
                        "path": os.path.join(self.nlst_dir, exam_str + ".pt"),
                        "y": y,
                        "y_seq": y_seq,
                        "y_mask": y_mask,
                        "time_at_event": time_at_event,
                        # lung_rads 0 indicates LungRads 1 and 2 (negative), 1 indicates LungRads 3 and 4 (positive)
                        # Follows "Pinsky PF, Gierada DS, Black W, et al: Performance of lung-RADS in the National Lung Screening Trial: A retrospective assessment. Ann Intern Med 162: 485-491, 2015"
                        "lung_rads": self.acc2lungrads[exam_int]
                    }

                    dataset.append(sample)

        if self.class_balance:
            # This data is highly imbalanced!
            # Introduce a method to deal with class imbalance (hint: think about your data loader)
            raise NotImplementedError("Not implemented yet")

        self.train = NLST_Dataset(self.train, self.train_transform, self.normalize, self.img_size, self.num_images)
        self.val = NLST_Dataset(self.val, self.test_transform, self.normalize, self.img_size, self.num_images)
        self.test = NLST_Dataset(self.test, self.test_transform, self.normalize, self.img_size, self.num_images)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.max_followup
        y_seq = np.zeros(self.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.max_followup - 1)
        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * (self.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class NLST_Dataset(torch.utils.data.Dataset):
    """
        Pytorch Dataset for NLST dataset. Loads preprocesses data from disk and applies data augmentation. Generates masks from bounding boxes stored in metadata..
    """

    def __init__(self, dataset, transforms, normalize, img_size=[256, 256], num_images=200):
        self.dataset = dataset
        self.transform = transforms
        self.normalize = normalize
        self.img_size = img_size
        self.num_images = num_images

        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample_path = self.dataset[idx]['path']
        sample = torch.load(sample_path)
        orig_pixel_spacing = torch.diag(torch.tensor(sample['pixel_spacing'] + [1]))
        num_slices = sample['x'].size()[0]

        right_side_cancer = sample['cancer_laterality'][0] == 1 and sample['cancer_laterality'][1] == 0
        left_side_cancer = sample['cancer_laterality'][1] == 1 and sample['cancer_laterality'][0] == 0

        # TODO: You can modify the data loading of the bounding boxes to suit your localization method.
        # Hint: You may want to use the "cancer_laterality" field to localize the cancer coarsely.

        if not sample['has_localization']:
            sample['bounding_boxes'] = None

        mask = self.get_scaled_annotation_mask(sample['bounding_boxes'], CACHE_IMG_SIZE + [num_slices])

        subject = tio.Subject( {
            'x': tio.ScalarImage(tensor=sample['x'].unsqueeze(0).to(torch.double), affine=orig_pixel_spacing),
            'mask': tio.LabelMap(tensor=mask.to(torch.double), affine=orig_pixel_spacing)
        })

        '''
            TorchIO will consistently apply the data augmentations to the image and mask, so that they are aligned. Note, the 'bounding_boxes' item will be wrong after after random transforms (e.g. rotations) in this implementation. 
        '''
        try:
            subject = self.transform(subject)
        except:
            raise Exception("Error with subject {}".format(sample_path))

        sample['x'], sample['mask'] = subject['x']['data'].to(torch.float), subject['mask']['data'].to(torch.float)
        ## Normalize volume to have 0 pixel mean and unit variance
        sample['x'] = self.normalize(sample['x'])

        ## Remove potentially none items for batch collation
        del sample['bounding_boxes']

        return sample

    def get_scaled_annotation_mask(self, bounding_boxes, img_size=[256, 256, 200]):
        """
        Construct bounding box masks for annotations.

        Args:
            - bounding_boxes: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1].
            - img_size per slice
        Returns:
            - mask of same size as input image, filled in where bounding box was drawn. If bounding_boxes = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
        """
        H, W, Z = img_size
        if bounding_boxes is None:
            return torch.zeros((1, Z, H, W))

        masks = []
        for slice in bounding_boxes:
            slice_annotations = slice["image_annotations"]
            slice_mask = np.zeros((H, W))

            if slice_annotations is None:
                masks.append(slice_mask)
                continue

            for annotation in slice_annotations:
                single_mask = np.zeros((H, W))
                x_left, y_top = annotation["x"] * W, annotation["y"] * H
                x_right, y_bottom = (
                    min( x_left + annotation["width"] * W, W-1),
                    min( y_top + annotation["height"] * H, H-1),
                )

                # pixels completely inside bounding box
                x_quant_left, y_quant_top = math.ceil(x_left), math.ceil(y_top)
                x_quant_right, y_quant_bottom = math.floor(x_right), math.floor(y_bottom)

                # excess area along edges
                dx_left = x_quant_left - x_left
                dx_right = x_right - x_quant_right
                dy_top = y_quant_top - y_top
                dy_bottom = y_bottom - y_quant_bottom

                # fill in corners first in case they are over-written later by greater true intersection
                # corners
                single_mask[math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
                single_mask[math.floor(y_top), x_quant_right] = dx_right * dy_top
                single_mask[y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
                single_mask[y_quant_bottom, x_quant_right] = dx_right * dy_bottom

                # edges
                single_mask[y_quant_top:y_quant_bottom, math.floor(x_left)] = dx_left
                single_mask[y_quant_top:y_quant_bottom, x_quant_right] = dx_right
                single_mask[math.floor(y_top), x_quant_left:x_quant_right] = dy_top
                single_mask[y_quant_bottom, x_quant_left:x_quant_right] = dy_bottom

                # completely inside
                single_mask[y_quant_top:y_quant_bottom, x_quant_left:x_quant_right] = 1

                # in case there are multiple boxes, add masks and divide by total later
                slice_mask += single_mask
                    
            masks.append(slice_mask)

        return torch.Tensor(np.array(masks)).unsqueeze(0)

    def get_summary_statement(self):
        num_patients = len(set([d['pid'] for d in self.dataset]))
        num_cancer = sum([d['y'] for d in self.dataset])
        num_cancer_year_1 = sum([d['y_seq'][0] for d in self.dataset])
        return "NLST Dataset. {} exams ({} with cancer in one year, {} cancer ever) from {} patients".format(len(self.dataset), num_cancer_year_1, num_cancer, num_patients)