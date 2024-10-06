from typing import Optional, List, Tuple, Type
from pathlib import Path
import timeit

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import tqdm

from nlos_navigation.ml import device, Model, Dataset
from nlos_navigation.utils import get_logger, BaseConfig, config_wrapper
from tbrn.utils.plotting import (
    plot_loss_curve,
    plot_points,
    plot_gaussians,
    plot_x_positions,
    plot_y_positions,
    plot_occupancy,
    plot_histogram_video,
)


@config_wrapper
class TrainerConfig(BaseConfig):
    """Settings for the training process. Used for type hinting.

    Attributes:
        outdir (Path): The directory to save the training results.
        plotssubdir (Path): The subdirectory to save the plots.

        batch_size (int): The batch size to use for training.
        learning_rate (float): The learning rate to use for training.
        epochs (int): The number of epochs to train for.

        instance (Type["Trainer"]): The trainer class to use.

        model (Model): The model to use for training.
        load_model (Optional[Path | bool]): The path to a model to load before
            training. If bool and is True, will load the model from the outdir.

        dataset (Dataset): The dataset to use for training.
    """

    outdir: Path
    plotssubdir: Path

    batch_size: int
    learning_rate: float
    epochs: int

    instance: Type["Trainer"]

    model: Model
    load_model: Optional[Path | bool] = None

    dataset: Dataset


class Trainer:
    """This is the trainer class for running training and evaluation.

    Args:
        config (TrainerConfig): The config to use for training and evaluation.
    """

    def __init__(self, config: TrainerConfig, *, criterion: nn.Module):
        self._config = config

        self._config.outdir.mkdir(parents=True, exist_ok=True)

        get_logger().info(f"Logging to {self._config.outdir / 'logs'}...")

        self._criterion = criterion

    def train(self):
        # Set to warn so we have something output to the error log
        get_logger().warning(f"Training the agent in {self._config.outdir}...")

        self._config.save(self._config.outdir / "config.yaml")
        self._config.pickle(self._config.outdir / "config.pkl")

        # Delete an existing finished file, if it exists
        if (finished := self._config.outdir / "finished").exists():
            finished.unlink()

        # Load the dataset
        # Split the dataset into training and evaluation
        dataset_size = len(self._config.dataset)
        indices = np.arange(dataset_size)
        train_split = np.random.choice(indices, int(dataset_size * 0.8), replace=False)
        eval_split = np.setdiff1d(indices, train_split)
        train_dataset = Subset(self._config.dataset, train_split)
        eval_dataset = Subset(self._config.dataset, eval_split)
        train_dataloader = DataLoader(train_dataset, self._config.batch_size, True)
        eval_dataloader = DataLoader(eval_dataset, self._config.batch_size, False)

        # Initialize the model, loss, and optimizer
        model = self._config.model.to(device)
        if self._config.load_model:
            load_model_path = (
                self._config.load_model
                if isinstance(self._config.load_model, Path)
                else self._config.outdir / "model.pth"
            )
            model.load_state_dict(torch.load(load_model_path, weights_only=True))
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)

        # Train
        losses, eval_losses = self.train_model(
            model,
            train_dataloader,
            eval_dataloader,
            self._criterion,
            optimizer,
            epochs=self._config.epochs,
        )

        # Save the trained model
        torch.save(model.state_dict(), self._config.outdir / "model.pth")

        # Plot the loss curves
        plotdir = self._config.outdir / self._config.plotssubdir
        plotdir.mkdir(parents=True, exist_ok=True)
        plot_loss_curve(losses, plotdir / "loss_curve.png")

        # Run an eval
        if len(eval_dataset) > 0:
            plot_loss_curve(eval_losses, plotdir / "eval_loss_curve.png")
            old_plotssubdir = self._config.plotssubdir
            with self._config.set_readonly_temporarily(False):
                self._config.plotssubdir = f"{self._config.plotssubdir}_eval"
                self.eval(eval_dataset)
                self._config.plotssubdir = old_plotssubdir

        # The finished file indicates to the evo script that the agent is done
        Path(self._config.outdir / "finished").touch()

    def eval(self, dataset: Optional[Dataset] = None):
        # Set to warn so we have something output to the error log
        get_logger().warning(f"Evaluating the agent in {self._config.outdir}...")

        self._config.save(self._config.outdir / "eval_config.yaml")

        # Load the model
        model_path = self._config.outdir / "model.pth"
        if self._config.load_model:
            model_path = (
                self._config.load_model
                if isinstance(self._config.load_model, Path)
                else self._config.outdir / "model.pth"
            )
        assert model_path.exists(), f"Model {model_path} does not exist."
        model = self._config.model.to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Run the full dataset first
        dataset = dataset or self._config.dataset
        assert len(dataset) > 0, "Dataset is empty."
        dataloader = DataLoader(dataset, 1, shuffle=False)

        # Evaluate
        losses, preds, gts = self.eval_model(model, dataloader)
        get_logger().info(f"Average loss: {sum(losses) / len(losses):.4f}")
        plotdir = self._config.outdir / self._config.plotssubdir
        self.plot_results(plotdir, preds, gts)

        # Loop through each dataset
        # if hasattr(dataset, "pkl_paths"):
        #     for i, pkl_path in enumerate(dataset.pkl_paths):
        #         dataset = dataset.copy(pkl_paths=[pkl_path])
        #         dataloader = DataLoader(dataset, 1, shuffle=False)

        #         # Evaluate
        #         losses, preds, gts = self.eval_model(model, dataloader)
        #         get_logger().info(f"Average loss {i}: {sum(losses) / len(losses):.4f}")
        #         plotdir = (
        #             self._config.outdir / self._config.plotssubdir / f"dataset_{i}"
        #         )
        #         self.plot_results(plotdir, preds, gts)

        # Save histograms as video
        # TODO (ay): Remove constants (3,3,128)
        histograms = np.array(
            [model.preprocess(hist).detach().numpy() for hist, _, _ in dataset]
        )
        histograms = histograms.reshape(-1, 3, 3, histograms.shape[-1])
        plot_histogram_video(histograms, plotdir)

    def speedtest(self):
        """This method will test a model to test inference speed."""
        # Load the model
        model_path = self._config.outdir / "model.pth"
        if self._config.load_model:
            model_path = (
                self._config.load_model
                if isinstance(self._config.load_model, Path)
                else self._config.outdir / "model.pth"
            )
        assert model_path.exists(), f"Model {model_path} does not exist."
        model = self._config.model.to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Load the dataset
        dataset = self._config.dataset
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Run the speed test
        def single_inference():
            for histograms, _ in dataloader:
                model(histograms)

        number = 100
        time = timeit.timeit(single_inference, number=number)
        print(f"Average inference time: {time / number / len(dataset):.4f} seconds")

    # ====================

    def train_model(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epochs: int = 100,
        eval_interval: int = 10,
        patience: int = 1000,
        min_delta: float = 0.001,
    ) -> Tuple[List[float], List[float]]:
        """Train the model with early stopping."""
        losses, eval_losses = [], []
        best_loss = float("inf")
        patience_counter = 0
        model.train()

        eval_loss = float("inf")
        with tqdm.tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                running_loss = 0.0
                for histograms, camera_pose, object_pose in train_dataloader:
                    outputs = model(histograms, camera_pose)
                    optimizer.zero_grad()
                    loss = criterion(outputs, object_pose)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_dataloader)

                # Log training loss
                get_logger().debug(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

                # Evaluate the model
                if (epoch + 1) % eval_interval == 0:
                    temp_eval_losses, _, _ = self.eval_model(model, eval_dataloader)
                    eval_loss = np.mean(temp_eval_losses)
                    eval_losses.append(eval_loss)
                    get_logger().info(
                        f"Epoch [{epoch+1}/{epochs}], "
                        f"Training Loss: {avg_loss:.4f}, "
                        f"Evaluation Loss: {eval_loss:.4f}"
                    )

                    # Early stopping check
                    if eval_loss < best_loss - min_delta:
                        best_loss = eval_loss
                        patience_counter = 0  # reset patience
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        get_logger().info(f"Early stopping at epoch {epoch+1}")
                        break

                # Update progress bar
                pbar.set_postfix(Loss=f"{avg_loss:.4f}", EvalLoss=f"{eval_loss:.4f}")
                pbar.update(1)

                losses.append(avg_loss)

        return losses, eval_losses

    def eval_model(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the model.

        Returns:
            losses (np.ndarray): The losses for each sample.
            predictions (np.ndarray): The predicted x, y coordinates.
            actuals (np.ndarray): The actual x, y coordinates.
        """
        losses, predictions, actuals = [], [], []
        model.eval()
        with torch.no_grad():
            for histograms, camera_pose, object_pose in dataloader:
                outputs = model(histograms, camera_pose)
                loss = self._criterion(outputs, object_pose)
                losses.append(loss.item())

                predicted, actual = outputs.cpu().numpy(), object_pose.cpu().numpy()
                model.debug(histograms)
                # get_logger().info(f"PREDICTED: {predicted}, ACTUAL: {actual}")
                predictions.extend(predicted)
                actuals.extend(actual)

        return np.array(losses), np.array(predictions), np.array(actuals)

    def plot_results(self, plotdir: Path, preds: np.ndarray, gts: np.ndarray):
        """Plot the results of the evaluation."""
        gts = np.squeeze(gts)
        preds = np.squeeze(preds)
        plotdir.mkdir(parents=True, exist_ok=True)
        if preds.shape[-1] == 3:
            plot_points(plotdir, gts[:, 0], gts[:, 1], preds[:, 0], preds[:, 1])
            plot_points(plotdir, gts[:, 0], gts[:, 1], preds[:, 0], preds[:, 1], True)
            # plot_gaussians(plotdir, gts[:, 0], gts[:, 1], preds[:, 0], preds[:, 1])
            plot_x_positions(plotdir, gts[:, 0], preds[:, 0])
            plot_y_positions(plotdir, gts[:, 1], preds[:, 1])
        else:
            plot_occupancy(plotdir, gts, preds)
        get_logger().info(f"Plots saved to {plotdir}")


class DummyTrainer(Trainer):
    """This is a dummy trainer class for testing purposes."""

    def train(self):
        get_logger().info("Training the dummy trainer")
        return 0.0

    def eval(self):
        get_logger().info("Evaluating the dummy trainer")
        return 0.0


if __name__ == "__main__":
    import argparse
    from tbrn.utils.config import run_hydra, Config

    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--test", action="store_true", help="Test the evo loop")

    def main(config: Config, *, train: bool, eval: bool) -> float:
        trainer = Trainer(config.trainer)

        if train:
            return trainer.train()
        elif eval:
            return trainer.eval()

    run_hydra(main, parser=parser)
