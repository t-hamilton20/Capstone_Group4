import torch


def convert_checkpoint(checkpoint_path, finalized_model_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    torch.save(state_dict, finalized_model_path)
    print(f"Model state dict saved to {finalized_model_path}")


checkpoint_path = './data/models/intermediates/4_VGG_Attacked.pth'
finalized_model_path = './data/models/VGG_Attacked_Final.pth'  # Desired path for the finalized model

# Convert the checkpoint
convert_checkpoint(checkpoint_path, finalized_model_path)
