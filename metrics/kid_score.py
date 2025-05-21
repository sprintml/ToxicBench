import os
import torch
import numpy as np
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

# Preprocessing images for InceptionV3
def preprocess_images(images, image_size=299):
    transform = Compose([
        Resize((image_size, image_size)),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(img) for img in images])

# Extract features using InceptionV3
def extract_features(images, model, device):
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features.cpu().numpy()

# Polynomial kernel for MMD
def polynomial_kernel(x, y, degree=3, coef0=1, gamma=None):
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    return (gamma * np.dot(x, y.T) + coef0) ** degree

# Compute KID with random sampling buckets
def compute_kid_with_buckets(x, y, kernel_func=polynomial_kernel, bucket_size=20, num_buckets=1):
    n_x, n_y = len(x), len(y)
    if n_x < bucket_size or n_y < bucket_size:
        raise ValueError("Bucket size must be smaller than the dataset size.")

    kid_scores = []
    for _ in range(num_buckets):
        x_sample = x[np.random.choice(n_x, bucket_size, replace=False)]
        y_sample = y[np.random.choice(n_y, bucket_size, replace=False)]

        kxx = kernel_func(x_sample, x_sample).mean()
        kyy = kernel_func(y_sample, y_sample).mean()
        kxy = kernel_func(x_sample, y_sample).mean()
        kid_scores.append(kxx + kyy - 2 * kxy)

    return np.mean(kid_scores), np.std(kid_scores)

# Run KID across multiple subfolders (buckets)
def evaluate_kid_for_folders(images_a, images_b, device='cuda', bucket_size=20, num_buckets=1):
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.fc = torch.nn.Identity()  # Use penultimate layer

    images_a = preprocess_images(images_a)
    images_b = preprocess_images(images_b)

    features_a = extract_features(images_a, model, device)
    features_b = extract_features(images_b, model, device)

    kid_mean, kid_std = compute_kid_with_buckets(
        features_a, features_b, bucket_size=bucket_size, num_buckets=num_buckets
    )

    results = {"mean": kid_mean, "std": kid_std}
    print(f"Score Mean = {kid_mean:.5f}, Std = {kid_std:.5f}")

    return results

def KIDScore(
        generated_images,
        original_images,
        device,
        num_samples_per_bucket=20,
        num_buckets=1,
) :

    results = evaluate_kid_for_folders(
        original_images,
        generated_images,
        device=device,
        bucket_size=num_samples_per_bucket,
        num_buckets=num_buckets
    )

    avg_mean, avg_std = results["mean"], results["std"]
    print(f"\nAggregate Score: Mean = {avg_mean:.5f}, Std = {avg_std:.5f}")

    return avg_mean
