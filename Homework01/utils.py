import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import accuracy_score


class Logger(ABC):
    """General interface for the loggers."""
 
    @abstractmethod
    def init(self, config: DictConfig) -> None:
        """Initialize the logger."""
 
    @abstractmethod
    def log(self, metrics: dict, step: int) -> None:
        """Log a dictionary containing the metrics at the current step."""
 
    @abstractmethod
    def finish(self) -> None:
        """Close the logger session."""

class WandbLogger(Logger):
    """Weights & Biases logger."""
 
    def __init__(self) -> None:
        self._run = None
 
    def init(self, config: DictConfig) -> None:
        import wandb
 
        cfg_dict = OmegaConf.to_container(config, resolve=True)
 
        self._run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.experiment_name,
            config=cfg_dict
        )
        print(f"[WandbLogger] Run initialized at: {self._run.url}")
 
    def log(self, metrics: dict, step: int) -> None:
        self._run.log(metrics, step=step)
 
    def finish(self) -> None:
        self._run.finish()
        print("[WandbLogger] Run closed.")

def unpack_loss(loss_output) -> tuple[torch.Tensor, dict]:
    """
    Loss output can be a sclar tensor or dict with the 'total' key
    Return (total_loss, dict with all the loss components).
    """

    if isinstance(loss_output, dict):
        if "total" in loss_output:
            total = loss_output["total"]
            components = loss_output
        else:
            # used by models like Faster R-CNN
            total = sum(loss_output.values())
            components = {**loss_output, "total": total}
    else:
        total = loss_output
        components = {"total": loss_output}
    return total, components


class Metric(ABC):
    """
    Define the general interface for the metrics.
    """
    @abstractmethod
    def accumulate(self, out, gts):
        pass

    @abstractmethod
    def compute(self) -> dict:
        pass

    @abstractmethod
    def reset(self):
        pass

class ClassificationMetric(Metric):
    """
    Class that computes the classification metrics.
    """
    def __init__(self):
        self.preds = []
        self.gts = []

    def accumulate(self, out, ys):
        preds = torch.argmax(out, dim=1).detach().cpu().numpy()
        self.preds.append(preds)
        self.gts.append(ys.detach().cpu().numpy())

    def compute(self) -> dict:
        preds = np.hstack(self.preds)
        gts = np.hstack(self.gts)
        return {
            "accuracy": accuracy_score(gts, preds),
        }

    def reset(self):
        self.preds = []
        self.gts = []

class ForwardPass(ABC):
    """
    Abstracts how the forward pass is executed and the loss is computed.

    Contract: always returns (out, loss_result)
      - out        : model predictions, None if not availables (es. training detection)
      - loss_result: scalar tensor, dict with key 'total', dict wo key 'total', or  None during eval
                     if the model doesn compute the loss internally
    """
    @abstractmethod
    def __call__(self, model, xs, gts) -> tuple:
        ...


class ExternalLossForward(ForwardPass):
    """
    Classic Models: out = model(x), loss = loss_fn(out, gts).
    Es: Classifiers.
    """
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, model, xs, gts):
        out = model(xs)
        loss_result = self.loss_fn(out, gts)
        return out, loss_result



class InternalLossForward(ForwardPass):
    """
    Models that compute the loss internally and return a dict containing the loss during training.
    Es: Faster R-CNN.
    - Training : model(xs, gts) -> dict containing the loss  (out non available)
    - Eval     : model(xs)      -> predictions (loss not computed)
    """
    def __call__(self, model, xs, gts):
        if model.training:
            loss_dict = model(xs, gts)   # {"loss_cls": ..., "loss_box": ..., ...}
            return None, loss_dict
        else:
            out = model(xs)
            return out, None

def to_device(data, device):
    """
    Moves data to the specified device (supports tensors, lists and dictionaries)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data

def evaluate_and_log(model, dl_val, device, epoch,
                      logger, metrics: Metric, forward_pass: ForwardPass):
    """
    Executes the evaluation loop, computes eval metrics and log them with training metrics.
    Metrics is a class that first accumulates all the batch results and then computes the metrics on the complete validation set

    """
    model.eval()
    val_loss_components = {}
    total_samples = 0

    metrics.reset()

    with torch.no_grad():
        for xs, gts in dl_val:
            xs = to_device(xs, device)
            gts = to_device(gts, device)
            batch_size = len(xs)

            out, loss_results = forward_pass(model, xs, gts)

            if loss_results is not None:
                _, components = unpack_loss(loss_results)

                for k, v in components.items():
                    val_loss_components.setdefault(k, 0.0)
                    val_loss_components[k] += (
                        v.item() if isinstance(v, torch.Tensor) else float(v)
                    ) * batch_size

            if out is not None:
                metrics.accumulate(out, gts)
            
            total_samples += batch_size

            

    val_metrics = {
        f"val/{k}": total / total_samples
        for k, total in val_loss_components.items()
    }

    val_metrics.update({f"val/{k}": v for k, v in metrics.compute().items()})

    logger.log({**val_metrics}, step=epoch)


def train_one_epoch(model, opt, dl_train, device, epoch, logger: Logger, forward_pass: ForwardPass):
    """
    Executes a training epoch

    """
    model.train()
    epoch_loss_components = {}
    total_samples = 0
    for xs, gts in dl_train:
        xs = to_device(xs, device)
        gts = to_device(gts, device)
        batch_size = len(xs)

        opt.zero_grad()

        _, loss_results = forward_pass(model, xs, gts)
        loss, components = unpack_loss(loss_results)
        loss.backward()
        opt.step()

            
        for k, v in components.items():
                epoch_loss_components.setdefault(k, 0.0)
                epoch_loss_components[k] += (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                ) * batch_size

        total_samples += batch_size
    
    # average loss for this epoch
    train_losses = {
        k: total / total_samples for k, total in epoch_loss_components.items()
    }

    print(f"Loss: {train_losses['total']:.4f}")

    train_losses["lr"] = opt.param_groups[0]["lr"]
    logger.log({f"train/{k}": v for k, v in train_losses.items()}, step=epoch)
       

def train_loop(model, opt, scheduler, dl_train, dl_val, forward_pass: ForwardPass, device,
                config: DictConfig, logger: Logger, metrics: Metric) -> None:
    
    """
    Complete training loop.
 
    Parameters extracted from `config`:
        epochs           -number of epochs
        log_every        -log every tot epochs
        experiment_name  - name of the experiment
    the loss function must compute the loss on the batch and use mean as the reduction
    """

    logger.init(config)

    epochs = config.training.epochs
    log_every = config.training.log_every

    for epoch in range(1, epochs+1):

        print(f"Epoch: {epoch}/{epochs}")
        train_one_epoch(model, opt, dl_train, device, epoch, logger, forward_pass)

        #logging
        if epoch % log_every == 0:
            evaluate_and_log(
                model, dl_val, device,
                epoch, logger, metrics, forward_pass
            )
        
        scheduler.step()

    logger.finish()


def save_classification_report(report: str, experiment_name: str,
    save_dir: str = "results") -> None:

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{experiment_name}.txt")

    with open(path, "w") as f:
        f.write(report)

    print(f"Report salvato in {path}")