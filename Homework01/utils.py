import numpy as np
import torch
import os
from abc import ABC, abstractmethod
from omegaconf import OmegaConf, DictConfig


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
            config=cfg_dict,
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

def evaluate_and_log(model: torch.nn.Module, dl_val, loss_function, device: torch.device, epoch: int,
                      train_losses: dict, logger: Logger, metrics_fn=None) -> dict:
    """
    Esegue la validation loop, raccoglie le metriche e le logga insieme
    a quelle di training.
    Metrics_fn deve essere una funzione che ritorna un dict con chiave metrica e
    valore il valore della metrica in float
 
    Ritorna il dict delle metriche di validation
    """
    model.eval()
    val_loss_components = {}
    val_extra_metrics = {}

    with torch.no_grad():
        for xs, gts in dl_val:
            xs = xs.to(device)
            gts = gts.to(device)

            out = model(xs)
            _, components = unpack_loss(loss_function(out, gts))

            for k, v in components.items():
                val_loss_components.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )

            # calcola metriche extra se la funzione è stata passata
            if metrics_fn is not None:
                extra = metrics_fn(out, gts) 
                for k, v in extra.items():
                    val_extra_metrics.setdefault(k, []).append(float(v))

    # media per ogni componente
    val_metrics = {
        f"val/{k}": float(np.mean(vs))
        for k, vs in {**val_loss_components, **val_extra_metrics}.items()
    }
    # prefissa le loss di training con "train/"
    prefixed_train = {f"train/{k}": v for k, v in train_losses.items()}

     
    logger.log({**prefixed_train, **val_metrics}, step=epoch)

    return val_metrics



def train_loop(model, opt, scheduler, loss_function, dl_train, dl_val, device,
                config: DictConfig, logger: Logger) -> None:
    
    """
    Loop di training completo.
 
    Parametri estratti da `config`:
        epochs           -numero di epoche
        log_every        -ogni quante epoche loggare
        experiment_name  - nome dell'esperimento
    """

    logger.init(config)

    epochs = config.epochs
    log_every = config.log_every

    for epoch in range(1, epochs+1):

        model.train()
        epoch_loss_components: dict[str, list] = {}

        for xs, gts in dl_train:

            xs.to(device)
            gts.to(device)

            opt.zero_grad()
            out = model(xs)

            loss, components = unpack_loss(loss_function(out, gts))
            loss.backward()
            opt.step()

            for k, v in components.items():
                epoch_loss_components.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else float(v)
                )
        
        # loss medie per questa epoca
        train_losses = {
            k: float(np.mean(vs)) for k, vs in epoch_loss_components.items()
        }
        
        #logga ogni log_everyy epoche
        if epoch % log_every == 0:
            evaluate_and_log(
                model, dl_val, loss_function, device,
                epoch, train_losses, logger,
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




