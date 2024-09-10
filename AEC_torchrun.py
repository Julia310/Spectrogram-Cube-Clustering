import torch
from torch.utils.data import Dataset, DataLoader
from ZarrDataLoader import ZarrDataset
from Models.networks import init_weights
from Models.ResNet_AEC import AEC
import torch.distributed as dist
from time import time
from torchvision import transforms

from torch.utils.data import random_split
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def flatten_batch(batch):
    # Convert list of samples (batch) into a tensor
    # Assuming each sample in the batch is a tensor of shape (mini_batch, channels, height, width)
    batch_tensor = torch.stack(batch,
                               dim=0)  # This creates a tensor of shape (batch_size, mini_batch, channels, height, width)

    # Flatten the batch and mini_batch dimensions
    batch_size, mini_batch, channels, height, width = batch_tensor.size()
    flattened_batch = batch_tensor.view(batch_size * mini_batch, channels, height, width)

    return flattened_batch


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            test_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
            early_stopping: bool = True,
            patience: int = 10,
            metrics: list = None
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_val_loss = float('inf')
        self.strikes = 0
        self.finished = False
        self.metrics = metrics if metrics is not None else [torch.nn.MSELoss(reduction='mean')]
        #if os.path.exists(snapshot_path):
        #    print("Loading snapshot")
        #    self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        start_batch = time()
        batch = batch.to(self.gpu_id)
        batch_size, mini_batch, channels, height, width = batch.size()
        batch = batch.view(batch_size * mini_batch, channels, height, width).to(self.gpu_id)
        self.optimizer.zero_grad()
        output = self.model(batch)
        #loss = F.mse_loss(output, batch)
        loss = self.metrics[0](output, batch)
        if loss.requires_grad:  # Check if loss requires gradients
            loss.backward()
            self.optimizer.step()
        #if self.gpu_id == 0:
        #    print(f"[GPU{self.gpu_id}] Batch processed: {time() - start_batch}")
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        running_loss = 0.0
        running_size = 0
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            start_time = time()
            with torch.set_grad_enabled(True):
                loss = self._run_batch(batch)
            running_loss += loss.item() * batch.size(0) * batch.size(1)
            running_size += batch.size(0) * batch.size(1)
            #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batch ({batch.shape}) processed in {batch_time:.4f} seconds")

        avg_epoch_loss = running_loss / running_size
        if self.gpu_id == 0:
            print(f"Epoch {epoch} | Average Loss: {avg_epoch_loss:.4f}")

    def _validate(self):
        self.model.eval()  # Set the model to evaluation mode
        running_loss = 0.0
        running_size = 0

        for batch in self.test_data:
            with torch.no_grad():  # No need to track gradients during validation
                loss = self._run_batch(batch)
            running_loss += loss.item() * batch.size(0) * batch.size(1)
            running_size += batch.size(0) * batch.size(1)

        # Convert running loss and size to tensors for all_reduce operation
        running_loss_tensor = torch.tensor([running_loss], device=self.gpu_id)
        running_size_tensor = torch.tensor([running_size], device=self.gpu_id)

        # Use dist.all_reduce to sum the losses and sizes from all GPUs
        dist.all_reduce(running_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_size_tensor, op=dist.ReduceOp.SUM)

        # Compute the average loss across all GPUs and samples
        avg_val_loss = running_loss_tensor.item() / running_size_tensor.item()

        if self.gpu_id == 0:
            print(f"Validation Loss: {avg_val_loss:.4e}")

        return avg_val_loss


    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            dist.barrier()

            epoch_val_loss = self._validate()

            # Early Stopping Logic
            if self.early_stopping:
                if epoch_val_loss < self.best_val_loss:
                    self.strikes = 0
                    self.best_val_loss = epoch_val_loss
                    # Saving the best model
                    best_model_path = 'Best_Model.pt'
                    torch.save(self.model.module.state_dict(), best_model_path)
                    if self.gpu_id == 0:
                        print(f"New best model saved with validation loss: {epoch_val_loss}")
                else:
                    self.strikes += 1

                if self.strikes > self.patience:
                    if self.gpu_id == 0:
                        print('Stopping early due to no improvement.')
                    self.finished = True
                    break  # Exit the training loop


def load_train_objs():
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        lambda x: x.double(),
    ])
    full_dataset = ZarrDataset('/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_5758.zarr', 4, transform=transform_pipeline)  # load your dataset
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset into training and test sets
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    model = AEC()
    model.apply(init_weights)
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, test_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=5,
        pin_memory=True,
        shuffle=False,
        #collate_fn=flatten_batch,
        sampler=DistributedSampler(dataset)
    )


@record
def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    train_set, test_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    test_data = prepare_dataloader(test_set, batch_size)

    trainer = Trainer(model, train_data, test_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=7, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)