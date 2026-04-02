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
        if "total" in loss_output:
            total = loss_output["total"]
            components = loss_output
        else:
            # modelli tipo Faster R-CNN che ritornano {"loss_cls": ..., "loss_box": ...}
            total = sum(loss_output.values())
            components = {**loss_output, "total": total}
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

class ForwardPass(ABC):
    """
    Astrae il modo in cui si esegue il forward pass e si calcola la loss.
    Contratto: ritorna sempre (out, loss_result)
      - out        : predizioni del modello, None se non disponibili (es. training detection)
      - loss_result: tensore scalare, dict con 'total', dict senza 'total', oppure None in eval
                     se il modello non calcola la loss internamente
    """
    @abstractmethod
    def __call__(self, model, xs, gts) -> tuple:
        ...


class ExternalLossForward(ForwardPass):
    """
    Modelli classici: out = model(x), loss = loss_fn(out, gts).
    Es: classificatori, segmentatori.
    """
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, model, xs, gts):
        out = model(xs)
        loss_result = self.loss_fn(out, gts)
        return out, loss_result



class InternalLossForward(ForwardPass):
    """
    Modelli che calcolano la loss internamente e ritornano un dict di loss in training.
    Es: Faster R-CNN, DETR, YOLOv5.
    - Training : model(xs, gts) -> dict di loss  (out non disponibile)
    - Eval     : model(xs)      -> predizioni     (loss non calcolata)
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
    Sposta ricorsivamente i dati sul device (supporta Tensori, Liste e Dizionari).
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
    
    # loss medie per questa epoca
    train_losses = {
        k: total / total_samples for k, total in epoch_loss_components.items()
    }

    print(f"Loss: {train_losses['total']:.4f}")

    train_losses["lr"] = opt.param_groups[0]["lr"]
    logger.log({f"train/{k}": v for k, v in train_losses.items()}, step=epoch)
       

def train_loop(model, opt, scheduler, dl_train, dl_val, forward_pass: ForwardPass, device,
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

        print(f"Epoch: {epoch}/{epochs}")
        train_one_epoch(model, opt, dl_train, device, epoch, logger, forward_pass)

        #logga ogni log_everyy epoche
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




