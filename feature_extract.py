import os
import glob
import argparse
from utils.embedding_generation import load_model, generate_umap, csv2h5ad, create_dataloader, extract_features
from sklearn.preprocessing import normalize
import torch

def generate_embeddings(args):

    print(torch.cuda.is_available())
    backbone = load_model(args.model_path, args.architecture, args.imagenet)
    backbone.eval()

    all_embeddings = []
    all_filenames = []
    sample_names = []

    print(f"Processing {args.save_path}...")
    # print(patch_path)
    dataloader = create_dataloader(os.path.dirname(args.save_path), args.num_workers)
    embeddings, filenames = extract_features(backbone, dataloader)
    print(filenames)
    all_embeddings.append(embeddings)
    all_filenames.extend(filenames)

    # scan_name = os.path.basename(os.path.dirname(root))
    sample_names.extend([s.split('__')[0] for s in filenames])

    all_embeddings = torch.cat(all_embeddings, 0)
    all_embeddings = normalize(all_embeddings)
    #print(sample_names)
    generate_umap(args.save_path,
                  args.experiment_name, all_embeddings, all_filenames, sample_names,args.quarter_path)

    csv2h5ad(args.save_path, args.experiment_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Feature extraction configuration')
    parser.add_argument('--save_path', type=str, default='test_set/sample',
                        help='directory of the patches')
    parser.add_argument('--architecture', type=str, default='resnet18',
                        help='Model architecture (default: resnet18)')
    parser.add_argument('--model_path', type=str, default='checkpoints_method_lambda0.9/simclr-epoch=46-train_total_loss=15.82.ckpt',
                        help='Path to model weights')
    parser.add_argument('--imagenet', type=bool,default=False)
    parser.add_argument('--quarter_path', type=str, default='mask_slides',
                        help='directory of saving quarters')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--neighbors', type=int, default=2,
                        help='Number of n_neighbors in UMAP')
    parser.add_argument('--experiment_name', type=str, default='test_neighbor2_3D_lambda0.9',
                        help='Name of the experiment')

    args = parser.parse_args()
    print(args)
    generate_embeddings(args)
    print('Script finished!')