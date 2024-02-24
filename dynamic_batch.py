import torch

class DynamicBatcher:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        images = []
        targets = []
        batch_size = self.batch_size
        prev_size = None  # Keep track of the size of the first image in the batch
        for img, tgt in self.dataset:
            if prev_size is None:
                prev_size = img.shape[1:]  # Assuming img is of shape (C, H, W)
            elif img.shape[1:] != prev_size:
                yield torch.stack(images), targets
                images = []
                targets = []
                prev_size = img.shape[1:]
                batch_size = self.batch_size  # Reset batch size for next batch
            
            images.append(img)
            targets.append(tgt)
            if len(images) == batch_size:
                yield torch.stack(images), targets
                images = []
                targets = []
                batch_size = self.batch_size  # Reset batch size for next batch

        # Yield the remaining data if any
        if len(images) > 0:
            yield torch.stack(images), targets
    
    def __len__(self):
        return len(self.batch_size)