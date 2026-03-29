import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig
from sklearn.metrics import accuracy_score


class Logger(ABC):
    """Interfaccia comune per i logger."""
 
    @abstractmethod
    def init(self, config: DictConfig) -> None:
        """Inizializza il logger."""
 
    @abstractmethod
    def log(self, metrics: dict, step: int) -> None:
        """Logga un dizionario di metriche allo step corrente."""
 
    @abstractmethod
    def finish(self) -> None:
        """Chiude la sessione di logging."""

class WandbLogger(Logger):
    """Logger che scrive su Weights & Biases."""
 
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
        print(f"[WandbLogger] Run inizializzata: {self._run.url}")
 
    def log(self, metrics: dict, step: int) -> None:
        self._run.log(metrics, step=step)
 
    def finish(self) -> None:
        self._run.finish()
        print("[WandbLogger] Run chiusa.")

def unpack_loss(loss_output) -> tuple[torch.Tensor, dict]:
    """
    La loss output puó essere un tensore scalare oopure un dict che ha la chiave 'total'
    Ritorna (loss_totale, dict_con_tutte_le_componenti).
    """

    if isinstance(loss_output, dict):
        total = loss_output["total"]
        components = loss_output
    else:
        total = loss_output
        components = {"total": loss_output}
    return total, components


class Metric(ABC):
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

def evaluate_and_log(model, dl_val, loss_function, device, epoch, train_losses,
                      logger, metrics: Metric):
    """
    Esegue il validation loop, raccoglie le metriche e le logga insieme
    a quelle di training.
    Metrics é una classe che accumula i risultati dei vari batch e poi calcola 
    la metriche su tutto il validation set

    """
    model.eval()
    val_loss_components = {}
    total_samples = 0

    metrics.reset()

    with torch.no_grad():
        for xs, gts in dl_val:
            xs = xs.to(device)
            gts = gts.to(device)
            batch_size = len(xs)

            out = model(xs)
            _, components = unpack_loss(loss_function(out, gts))

            for k, v in components.items():
                val_loss_components.setdefault(k, 0.0)
                val_loss_components[k] += (
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                ) * batch_size

            total_samples += batch_size

            metrics.accumulate(out, gts)

    val_metrics = {
        f"val/{k}": total / total_samples
        for k, total in val_loss_components.items()
    }

    val_metrics.update({f"val/{k}": v for k, v in metrics.compute().items()})

    logger.log({**val_metrics}, step=epoch)


def train_loop(model, opt, scheduler, loss_function, dl_train, dl_val, device,
                config: DictConfig, logger: Logger, metrics: Metric) -> None:
    
    """
    Loop di training completo.
 
    Parametri estratti da `config`:
        epochs           -numero di epoche
        log_every        -ogni quante epoche loggare
        experiment_name  - nome dell'esperimento

    la loss function deve calcolare la loss sul batch ed usare come reduce mean
    """

    logger.init(config)

    epochs = config.training.epochs
    log_every = config.training.log_every

    for epoch in range(1, epochs+1):

        model.train()
        epoch_loss_components = {}
        total_samples = 0

        for xs, gts in dl_train:

            xs = xs.to(device)
            gts = gts.to(device)
            batch_size = len(xs)

            opt.zero_grad()
            out = model(xs)

            loss, components = unpack_loss(loss_function(out, gts))
            loss.backward()
            opt.step()

            
            for k, v in components.items():
                    epoch_loss_components.setdefault(k, 0.0)
                    epoch_loss_components[k] += (
                        v.item() if isinstance(v, torch.Tensor) else float(v)
                    ) * batch_size

            total_samples += batch_size
        
        # loss medie per questa epoca
        train_losses = {
            k: total / total_samples for k, total in epoch_loss_components.items()
        }

        print(f"Epoch {epoch}/{epochs} - loss: {train_losses['total']:.4f}")

        train_losses["lr"] = opt.param_groups[0]["lr"]
        logger.log({f"train/{k}": v for k, v in train_losses.items()}, step=epoch)

        #logga ogni log_everyy epoche
        if epoch % log_every == 0:
            evaluate_and_log(
                model, dl_val, loss_function, device,
                epoch, train_losses, logger, metrics
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




