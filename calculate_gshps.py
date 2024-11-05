import os
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse


class SingleFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(SingleFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        filename = os.path.basename(path)
        tuple_with_path = (original_tuple + (path, filename))
        return tuple_with_path

    def find_classes(self, directory: str):
        classes = [directory]
        class_to_idx = {directory: 0}
        return classes, class_to_idx


def get_feature_vector(device, model, loader):
    feature_vectors = []
    all_file_names = ()
    with torch.no_grad():
        for images, labels, paths, file_names in loader:
            images = images.to(device)
            features = model(images)
            features = features.view(features.size(0), -1)
            feature_vectors.append(features.cpu())
            all_file_names += file_names
    avg_feature_vector = torch.mean(torch.cat(feature_vectors, dim=0), dim=0)
    return avg_feature_vector, all_file_names


def compute_gshps(ref_image_path, gen_image_path, batch_size, classification_model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = torchvision.models.VGG16_Weights.DEFAULT.transforms()

    model = torch.load(classification_model)
    print(f"Loaded classification model from {classification_model}")
    model = model.features
    model.to(device)
    model.eval()

    ref_dataset = SingleFolderWithPaths(root=ref_image_path, transform=transform)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    avg_feature_vector_ref, all_file_names_ref = get_feature_vector(device, model, ref_loader)

    gen_dataset = SingleFolderWithPaths(root=gen_image_path, transform=transform)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    avg_feature_vector_gen, all_file_names_gen = get_feature_vector(device, model, gen_loader)

    if len(all_file_names_ref) != len(all_file_names_gen):
        print(f"Reference images: {len(all_file_names_ref)}, Generated images: {len(all_file_names_gen)}")
        raise ValueError("The number of images in the reference and generated folders do not match.")

    if list(all_file_names_ref) != list(all_file_names_gen):
        print("The filenames in the reference and generated folders do not match or are not in the same order.")
        raise ValueError("Filenames mismatch between reference and generated images.")

    average_gshps = nn.functional.cosine_similarity(avg_feature_vector_ref, avg_feature_vector_gen, dim=0).item()

    print(f"GSHPS: {average_gshps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute GSHPS metric between reference and generated images.")
    parser.add_argument("--ref", type=str, help="Path to the reference images folder.")
    parser.add_argument("--gen", type=str, help="Path to the generated images folder.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading.")
    parser.add_argument("--classification_model", type=str, default="gshps_vgg16.pt",
                        help="Path to the classification model.")
    args = parser.parse_args()
    compute_gshps(args.ref, args.gen, args.batch_size, args.classification_model)
