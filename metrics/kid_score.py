"""Compute KID Score for generated images/original images comparison."""

import torch
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def preprocess_images(images, image_size=299):
    transform = Compose([
        Resize((image_size, image_size)),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(img) for img in images])


def KIDScore(
        generated_images,
        original_images,
        num_samples_per_bucket=20,
        num_buckets=1
) :
    
    metric = KernelInceptionDistance(
        subsets=num_buckets,
        subset_size=num_samples_per_bucket,
        return_pred=False
    )

    # preprocess of input images
    generated_tensor = preprocess_images(images=generated_images)
    original_tensor = preprocess_images(images=original_images)

    metric.update(generated_tensor, real=False)
    metric.update(original_tensor, real=True)

    kid_score = metric.compute()

    return kid_score[0]
