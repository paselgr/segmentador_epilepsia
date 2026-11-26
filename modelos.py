import os

import matplotlib.pyplot as plt
from monai.inferers import SlidingWindowInferer
import pennylane as qml
from pennylane import numpy as pnp
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR


class DiceLoss(nn.Module):
    """
    Calculation of the Dice coefficient evaluated on each sample
    individually and returning the mean across all of them
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: Model output (logits)       -> [Batch, 1, D, H, W]
        # targets: Ground truth mask (0 or 1) -> [Batch, 1, D, H, W]
        
        # 1. Apply Sigmoid to get probabilities between 0 and 1
        inputs = torch.sigmoid(inputs).float()
        targets = (targets > 0).float()
        
        # 2. Flatten tensors to compute the intersection easily
        assert inputs.shape == targets.shape, 'Input and target dimensions do not match'
        inputs  = torch.flatten(inputs, start_dim=1)
        targets = torch.flatten(targets, start_dim=1)
        
        # 3. Compute Dice
        intersection = (inputs * targets).sum(dim=1)
        dice_tensor = (2. * intersection + self.smooth) / (inputs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        dice = torch.mean(dice_tensor)
        
        # 4. Return Loss (1 - Dice)
        return 1 - dice


class ConvBlock(nn.Module):
    """
    Basic convolution block
    Conv3d -> InstanceNorm3d -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope, stride=1):
        super().__init__()
        # Padding calculated to preserve dimensions with kernel 3 (p=1)
        padding = 1 if kernel_size == 3 else 0
        
        self.conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True,
            #padding_mode='replicate' # Peak in memory usage
        )
        self.norm = nn.InstanceNorm3d(
            out_channels, 
            eps=1e-05, 
            affine=True
        )
        self.act = nn.LeakyReLU(inplace=True, negative_slope=negative_slope)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class StackedConvLayers(nn.Module):
    """
    Groups convolutions.
    The first convolution handles the stride (downsampling)
    """
    def __init__(self, in_channels, out_channels, kernel_size, negative_slope, stride, num_convs):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        # First convolution applies the stride and changes the channels
        self.blocks.append(ConvBlock(in_channels, out_channels, kernel_size, negative_slope, stride))
        
        # Subsequent convolutions preserve the dimensions
        for _ in range(num_convs - 1):
            self.blocks.append(ConvBlock(out_channels, out_channels, kernel_size, negative_slope, stride=1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class NNUNet3D_Clasico(pl.LightningModule):
    """Classical model based on nnU-Net"""
    def __init__(self, input_channels=1, num_classes=1, learning_rate=1e-2, negative_slope=0.02):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate

        filters = [32, 64, 128, 256, 320, 320]
        strides = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,1]]
        n_conv_encoder = [2, 2, 2, 2, 2, 2]
        n_conv_decoder = [2, 2, 2, 2, 2]
        kernel_size = 3

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        current_in_channels = input_channels
        
        for (stride, n_convs, out_feats) in zip(strides, n_conv_encoder, filters):
            
            block = StackedConvLayers(
                current_in_channels, 
                out_feats, 
                kernel_size,
                negative_slope, 
                stride, 
                n_convs
            )
            self.encoder_blocks.append(block)
            current_in_channels = out_feats

        # --- Decoder ---
        # The decoder goes from bottom to top (reversing lists)
        self.decoder_blocks = nn.ModuleList()
        self.transp_convs = nn.ModuleList()
        
        # Decoder filters (excluding the deepest bottleneck): [320, 256, 128, 64, 32]
        decoder_filters = filters[:-1][::-1]
        # Decoder strides (corresponding to encoder strides to revert)
        # Skip the first encoder stride [1,1,1] since there is no final resolution upsampling
        decoder_strides = strides[1:][::-1]
        
        # Current input comes from the bottleneck (last encoder block) = 320
        current_in_channels = filters[-1]

        for (stride, out_feats, num_convs) in zip(decoder_strides, decoder_filters, n_conv_decoder):
            kernel_map  = {1: 1, 2: 4}
            padding_map = {1: 0, 2: 1}
            kernel_size_transp = tuple(kernel_map[dim]  for dim in stride)
            padding_transp     = tuple(padding_map[dim] for dim in stride)

            # Transposed convolution for upsampling
            self.transp_convs.append(
                nn.ConvTranspose3d(
                    current_in_channels,
                    out_feats, 
                    kernel_size=kernel_size_transp,
                    stride=stride,
                    bias=False,
                    padding=padding_transp
                )
            )
            
            # Convolution block after concatenation
            # Input will be: out_feats (from upsample) + out_feats (from skip connection)
            self.decoder_blocks.append(
                StackedConvLayers(
                    out_feats * 2, # Concatenation
                    out_feats, 
                    kernel_size,
                    negative_slope,
                    num_convs=num_convs,
                    stride=1
                )
            )
            current_in_channels = out_feats

        # --- Segmentation Head (1x1x1 Conv) ---
        self.seg_head = nn.Conv3d(filters[0], num_classes, kernel_size=1, bias=False)

        self._initialize_weights(negative_slope=negative_slope)

    def _initialize_weights(self, negative_slope):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=negative_slope)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = []
        
        # Encoder Path
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
        
        # The last item in skips is the bottleneck; it is not used as a skip connection for itself
        x = skips.pop()
        
        # Decoder Path
        for (transp, block) in zip(self.transp_convs, self.decoder_blocks):
            skip = skips.pop() # Obtener skip connection correspondiente
            
            # Upsample
            x = transp(x)
            
            # Handle minor size mismatches due to padding/odd strides
            if x.shape[2:] != skip.shape[2:]:
                x = interpolate(x, size=skip.shape[2:], mode='nearest')
            
            # Concatenate
            x = torch.cat([x, skip], dim=1)
            
            # Convolutions
            x = block(x)
            
        return self.seg_head(x)


class SegmentationModule_Clasico(pl.LightningModule):
    """
    Specify what happens before and after
    training, validation, and test
    """
    def __init__(self, pytorch_model, max_epochs, learning_rate=1e-2):
        super().__init__()
        self.model = pytorch_model
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
        # Define loss and metrics
        self.criterion = DiceLoss(smooth=1e-5)

        self.test_step_outputs = []

    def forward(self, x):
        # Inference step
        return self.model(x)

    def _calculate_dice_score(self, logits, targets):
        """Helper function for metric (not for gradients)"""
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float() # Binarize prediction

        preds   = torch.flatten(preds, start_dim=1)
        targets = torch.flatten(targets, start_dim=1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        # Dice coefficient
        dice_tensor = (2. * intersection + 1e-5) / (union + 1e-5)
        dice = torch.mean(dice_tensor)
        return dice

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        masks = batch["labels"]
        
        # Forward pass
        outputs = self(inputs) 
        
        # Compute loss and metrics
        loss = self.criterion(outputs, masks)
        dice_score = self._calculate_dice_score(outputs, masks)
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("train_dice", dice_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))

        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        masks = batch["labels"]
        
        outputs = self(inputs)

        loss = self.criterion(outputs, masks)
        dice_score = self._calculate_dice_score(outputs, masks)
        
        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("val_dice", dice_score, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        return loss
    
    def _calculate_individual_dice_scores(self, outputs, targets):
        """Computes the Dice Score for each individual item in the batch"""
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        # Flatten spatial dimensions: [B, C, D, H, W] -> [B, -1]
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice_per_sample = (2. * intersection + 1e-5) / (union + 1e-5)
        return dice_per_sample
    
    def test_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        masks = batch["labels"]
        
        outputs = self(inputs)

        # Apply SlidingWindowInferer to compute the Dice coefficient on whole scans
        patch_size = (112, 131, 98)
        sw_batch_size = 4
        overlap = 0.9
        sliding_window = SlidingWindowInferer(\
            roi_size=patch_size, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian")
        outputs_sliding_window = sliding_window(inputs, self.model)

        loss = self.criterion(outputs, masks)
        individual_dices = self._calculate_individual_dice_scores(outputs_sliding_window, masks)
        
        # Store in list (move to CPU/numpy to avoid GPU memory usage)
        self.test_step_outputs.append(individual_dices.detach().cpu())

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        # Concatenate all collected results
        all_dices = torch.cat(self.test_step_outputs)
        version = self.logger.version
        ruta_carpeta = os.path.join('resultados_enteros', f'version_{version}')
        os.makedirs(ruta_carpeta, exist_ok=True)
        nombre_archivo = os.path.join(ruta_carpeta, 'resultados_enteros.txt')
        # Save all Dice coefficient values
        with open(nombre_archivo, "w") as f:
            f.write(f"Resultados de test - Version {version}\n")
            for test_dice in all_dices:
                f.write(f"{test_dice}\n")

        # Compute global mean, standard deviation,
        # median, and interquartile range
        avg_dice = torch.mean(all_dices)
        std_dice = torch.std(all_dices)
        med_dice = torch.median(all_dices)
        iqr_dice = torch.quantile(all_dices, 0.75) - torch.quantile(all_dices, 0.25)

        print(f"\n--- Final Test Results ---")
        print(f"Mean Dice: {avg_dice:.4f}")
        print(f"Standard Deviation Dice: {std_dice:.4f}")
        print(f"Median Dice: {med_dice:.4f}")
        print(f"Interquartile Range Dice: {iqr_dice:.4f}")

        # Log final metrics
        self.log("test_dice_avg", avg_dice)
        self.log("test_dice_std", std_dice)
        self.log("test_dice_med", med_dice)
        self.log("test_dice_iqr", iqr_dice)

        # Clear the list for future tests or validations
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.99,
            weight_decay=3e-05,
            nesterov=True
        )

        # Define the scheduler function
        # Formula is (1 - epoch / max_epochs) ** 0.9
        scheduler = PolynomialLR(
            optimizer,
            total_iters=self.max_epochs,
            power=0.9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", # Updated every epoch
            },
        }


class PlotCallback_Clasico(pl.Callback):
    """
    Produce a cumulative plot of the Dice coefficient
    and Loss at the end of each epoch.
    Plots for the same version are overwritten
    so only the latest one is kept.
    """
    def __init__(self):
        super().__init__()
        self.base_save_dir = 'graficas'
        # Diccionarios para mantener el histórico completo
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': []
        }
        self.save_dir = None

    def on_train_start(self, trainer, pl_module):
        # Access the logger path (ej. lightning_logs/version_0)
        if trainer.logger:
            version_dir = trainer.logger.version
            if isinstance(version_dir, int):
                version_dir = f"version_{version_dir}"
            # Build the dynamic path
            self.save_dir = os.path.join(self.base_save_dir, version_dir)
        else:
            self.save_dir = self.base_save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve metrics from the trainer
        metrics = trainer.callback_metrics
        
        t_loss = metrics.get('train_loss')
        v_loss = metrics.get('val_loss')
        t_dice = metrics.get('train_dice')
        v_dice = metrics.get('val_dice')

        # Append only if the value exists in this epoch
        if t_loss is not None:
            self.history['train_loss'].append(t_loss.item())
        if v_loss is not None:
            self.history['val_loss'].append(v_loss.item())
        if t_dice is not None:
            self.history['train_dice'].append(t_dice.item())
        if v_dice is not None:
            self.history['val_dice'].append(v_dice.item())

        # Generate the cumulative plot
        self._generate_loss_plot(trainer.current_epoch)
        self._generate_dice_plot(trainer.current_epoch)

    def _generate_loss_plot(self, epoch):
        plt.figure(figsize=(10, 6))
        
        epochs_range = range(len(self.history['train_loss']))

        if self.history['train_loss']:
            plt.plot(epochs_range, self.history['train_loss'], label='Train Loss')
        
        if self.history['val_loss']:
            # Adjust range if val_loss has fewer entries (e.g., if not validating every epoch)
            val_range = range(len(self.history['val_loss']))
            plt.plot(val_range, self.history['val_loss'], label='Val Loss')

        plt.title(f'Loss del modelo por época')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida (Loss)')
        plt.legend()
        plt.grid(True, alpha=0.7)

        # Save the image
        plt.savefig(os.path.join(self.save_dir, f"loss_grafica.png"))
        plt.close()

    def _generate_dice_plot(self, epoch):
        plt.figure(figsize=(10, 6))
        
        epochs_range = range(len(self.history['train_dice']))

        if self.history['train_dice']:
            plt.plot(epochs_range, self.history['train_dice'], label='Train Dice')
        
        if self.history['val_dice']:
            # Adjust range if val_loss has fewer entries (e.g., if not validating every epoch)
            val_range = range(len(self.history['val_dice']))
            plt.plot(val_range, self.history['val_dice'], label='Val Dice')

        plt.title(f'Coeficiente de Dice por época')
        plt.xlabel('Épocas')
        plt.ylabel('Coeficiente de Dice')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.7)

        # Save the image
        plt.savefig(os.path.join(self.save_dir, f"dice_grafica.png"))
        plt.close()


class QuantumBridge(nn.Module):
    """
    Quantum bridge used in the bottleneck
    of the hybrid model.
    """
    def __init__(self, channels, num_qubits=8, num_layers=3):
        super().__init__()
        self.n_qubits = num_qubits
        self.num_layers = num_layers
        
        # Create dynamic device based on the number of qubits received
        dev = qml.device("default.qubit", wires=self.n_qubits)

        # Apply angle embedding to the data and entangle
        # the qubits using StronglyEntanglingLayers
        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Channel reduction (from 320 to n_qubits)
        self.reduce = nn.Conv3d(channels, self.n_qubits, kernel_size=1)
        
        # PennyLane Quantum Layer
        weight_shapes = {"weights": (self.num_layers, self.n_qubits, 3)} 
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Expansion (from n_qubits to 320)
        self.expand = nn.Conv3d(self.n_qubits, channels, kernel_size=1)
        
        # Learnable scaling parameter for quantum influence (Quantum Gamma)
        self.gamma = nn.Parameter(torch.tensor([0.5])) 

    def forward(self, x):
        identity = x 
        
        q_in = self.reduce(x) 
        b, c, d, h, w = q_in.shape
        
        # Reorder dimensions to apply the PennyLane function
        q_in = q_in.permute(0, 2, 3, 4, 1).reshape(-1, self.n_qubits)
        q_in = torch.tanh(q_in) * pnp.pi 
        
        q_out = self.q_layer(q_in)
        
        q_out = q_out.reshape(b, d, h, w, self.n_qubits).permute(0, 4, 1, 2, 3)
        out = self.expand(q_out)
        
        # Convex linear combination of the classical and quantum parts
        clamped_gamma = torch.clamp(self.gamma, min=0.0, max=1.0)
        return (1 - clamped_gamma) * identity + clamped_gamma * out


class NNUNet3D_Cuantica(pl.LightningModule):
    """Hybrid model based on nnU-Net"""
    def __init__(self, input_channels=1, num_classes=1, learning_rate=1e-2, negative_slope=0.02, number_quantum_layers=3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.number_quantum_layers = number_quantum_layers

        filters = [32, 64, 128, 256, 320, 320]
        strides = [[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,1]]
        n_conv_encoder = [2, 2, 2, 2, 2, 2]
        n_conv_decoder = [2, 2, 2, 2, 2]
        kernel_size = 3

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        current_in_channels = input_channels
        
        for (stride, n_convs, out_feats) in zip(strides, n_conv_encoder, filters):
            
            block = StackedConvLayers(
                current_in_channels, 
                out_feats, 
                kernel_size,
                negative_slope, 
                stride, 
                n_convs
            )
            self.encoder_blocks.append(block)
            current_in_channels = out_feats

        # --- Decoder ---
        # The decoder goes from bottom to top (reversing lists)
        self.decoder_blocks = nn.ModuleList()
        self.transp_convs = nn.ModuleList()
        
        # Decoder filters (excluding the deepest bottleneck): [320, 256, 128, 64, 32]
        decoder_filters = filters[:-1][::-1]
        # Decoder strides (corresponding to encoder strides to revert)
        # Skip the first encoder stride [1,1,1] since there is no final resolution upsampling
        decoder_strides = strides[1:][::-1]
        
        # Current input comes from the bottleneck (last encoder block) = 320
        current_in_channels = filters[-1]

        for (stride, out_feats, num_convs) in zip(decoder_strides, decoder_filters, n_conv_decoder):
            kernel_map  = {1: 1, 2: 4}
            padding_map = {1: 0, 2: 1}
            kernel_size_transp = tuple(kernel_map[dim]  for dim in stride)
            padding_transp     = tuple(padding_map[dim] for dim in stride)

            # Transposed convolution for upsampling
            self.transp_convs.append(
                nn.ConvTranspose3d(
                    current_in_channels,
                    out_feats, 
                    kernel_size=kernel_size_transp,
                    stride=stride,
                    bias=False,
                    padding=padding_transp
                )
            )
            
            # Convolution block after concatenation
            # Input will be: out_feats (from upsample) + out_feats (from skip connection)
            self.decoder_blocks.append(
                StackedConvLayers(
                    out_feats * 2, # Concatenation
                    out_feats, 
                    kernel_size,
                    negative_slope,
                    num_convs=num_convs,
                    stride=1
                )
            )
            current_in_channels = out_feats

        # Quantum Bridge in the Bottleneck
        # The number of filters in the bridge is filters[-1], i.e., 320.
        self.quantum_bridge = QuantumBridge(filters[-1], num_qubits=8, num_layers=self.number_quantum_layers)

        # --- Segmentation Head (1x1x1 Conv) ---
        self.seg_head = nn.Conv3d(filters[0], num_classes, kernel_size=1, bias=False)

        self._initialize_weights(negative_slope=negative_slope)

    def _initialize_weights(self, negative_slope):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', a=negative_slope)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skips = []
        
        # Encoder Path
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)
        
        # The last item in skips is the bottleneck; it is not used as a skip connection for itself
        x = skips.pop()

        # Apply the quantum bridge here, where dimensions are smallest
        x = self.quantum_bridge(x)
        
        # Decoder Path
        for (transp, block) in zip(self.transp_convs, self.decoder_blocks):
            skip = skips.pop() # Obtener skip connection correspondiente (FILO)
            
            # Upsample
            x = transp(x)
            
            # Handle minor size mismatches due to padding/odd strides
            if x.shape[2:] != skip.shape[2:]:
                x = interpolate(x, size=skip.shape[2:], mode='nearest')
            
            # Concatenate
            x = torch.cat([x, skip], dim=1)
            
            # Convolutions
            x = block(x)
            
        return self.seg_head(x)


class SegmentationModule_Cuantica(pl.LightningModule):
    """
    Specify what happens before and after
    training, validation, and test
    """
    def __init__(self, pytorch_model, max_epochs, learning_rate=1e-3):
        super().__init__()
        self.model = pytorch_model
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        
        # Define the loss and metrics
        self.criterion = DiceLoss(smooth=1e-5)

        self.test_step_outputs = []

    def forward(self, x):
        # The inference step
        return self.model(x)

    def _calculate_dice_score(self, logits, targets, smooth=1e-5):
        """Función auxiliar para métrica (no para gradiente)"""
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float() # Binarize prediction

        preds   = torch.flatten(preds, start_dim=1)
        targets = torch.flatten(targets, start_dim=1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        # Dice coefficient
        dice_tensor = (2. * intersection + smooth) / (union + smooth)
        dice = torch.mean(dice_tensor)
        return dice

    def training_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        masks = batch["labels"]
        
        # Forward pass
        outputs = self(inputs) 
        
        # Calculate loss and metrics
        loss = self.criterion(outputs, masks)
        dice_score = self._calculate_dice_score(outputs, masks)
        
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("train_dice", dice_score, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))

        # Log the absolute value of quantum gamma
        # (proportion of the quantum layer used) to see its importance
        q_clamped_weight = torch.clamp(self.model.quantum_bridge.gamma, min=0.0, max=1.0).item() 
        self.log("gamma_weight", q_clamped_weight, on_step=False, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))

        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        masks = batch["labels"]
        
        outputs = self(inputs)

        loss = self.criterion(outputs, masks)
        dice_score = self._calculate_dice_score(outputs, masks)
        
        # Log validation metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        self.log("val_dice", dice_score, on_epoch=True, prog_bar=True, batch_size=inputs.size(0))
        return loss
    
    def _calculate_individual_dice_scores(self, outputs, targets, smooth=1e-5):
        """Calcula el Dice Score por cada elemento individual en el batch"""
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        # Flatten spatial dimensions: [B, C, D, H, W] -> [B, -1]
        preds = preds.view(preds.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        intersection = (preds * targets).sum(dim=1)
        union = preds.sum(dim=1) + targets.sum(dim=1)

        dice_per_sample = (2. * intersection + smooth) / (union + smooth)
        return dice_per_sample
    
    def test_step(self, batch, batch_idx):
        inputs = batch["pixel_values"]
        masks = batch["labels"]
        
        outputs = self(inputs)

        # Apply SlidingWindowInferer to calculate
        # the Dice coefficient on full scans
        patch_size = (112, 131, 98)
        sw_batch_size = 4
        overlap = 0.9
        sliding_window = SlidingWindowInferer(\
            roi_size=patch_size, sw_batch_size=sw_batch_size, overlap=overlap, mode="gaussian")
        outputs_sliding_window = sliding_window(inputs, self.model)


        loss = self.criterion(outputs, masks)
        individual_dices = self._calculate_individual_dice_scores(outputs_sliding_window, masks)
        
        # Save to the list (convert to CPU/numpy to avoid GPU memory usage)
        self.test_step_outputs.append(individual_dices.detach().cpu())

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        # Concatenate all collected results
        all_dices = torch.cat(self.test_step_outputs)
        version = self.logger.version
        ruta_carpeta = os.path.join('resultados_enteros', f'version_{version}')
        os.makedirs(ruta_carpeta, exist_ok=True)
        nombre_archivo = os.path.join(ruta_carpeta, 'resultados_enteros.txt')
        with open(nombre_archivo, "w") as f: # Save all Dice coefficient values
            f.write(f"Resultados de test - Version {version}\n")
            for test_dice in all_dices:
                f.write(f"{test_dice}\n")

        # Compute global mean, standard deviation,
        # median, and interquartile range
        avg_dice = torch.mean(all_dices)
        std_dice = torch.std(all_dices)
        med_dice = torch.median(all_dices)
        iqr_dice = torch.quantile(all_dices, 0.75) - torch.quantile(all_dices, 0.25)

        print(f"\n--- Final Test Results ---")
        print(f"Mean Dice: {avg_dice:.4f}")
        print(f"Standard Deviation Dice: {std_dice:.4f}")
        print(f"Median Dice: {med_dice:.4f}")
        print(f"Interquartile Range Dice: {iqr_dice:.4f}")

        # Log final metrics
        self.log("test_dice_avg", avg_dice)
        self.log("test_dice_std", std_dice)
        self.log("test_dice_med", med_dice)
        self.log("test_dice_iqr", iqr_dice)

        # Clear the list for future tests or validations
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        # Separate parameters into Classical and Quantum
        # In our case we will not make a distinction between them,
        # but we keep the configuration separate in case you want
        # to run experiments changing lr, weight_decay...
        bridge_bottleneck = self.model.quantum_bridge
        
        quantum_params = list(bridge_bottleneck.parameters())
        quantum_ids = set(id(p) for p in quantum_params)
        classical_params = [p for p in self.parameters() if id(p) not in quantum_ids]

        # Define the optimizer with parameter groups
        optimizer = SGD([
            {
                "params": classical_params, 
                "name": "classical"
            },
            {
                "params": quantum_params, 
                "name": "quantum",
            }
        ], 
        lr=self.learning_rate, 
        momentum=0.9, 
        weight_decay=3e-05, 
        nesterov=True
        )

        # Define the scheduler function
        # The formula is (1 - epoch / max_epochs) ** 0.9
        scheduler = PolynomialLR(
            optimizer,
            total_iters=self.max_epochs,
            power=0.9
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class PlotCallback_Cuantica(pl.Callback):
    """
        Produce a cumulative plot of the Dice coefficient,
        Loss, and quantum gamma at the end of each epoch.
        Plots for the same version are overwritten
        so only the latest one is kept.
        """
    def __init__(self):
        super().__init__()
        self.base_save_dir = 'graficas'
        # Dictionaries to keep the full history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],

            'gamma_cuantica': []
        }
        self.save_dir = None

    def on_train_start(self, trainer, pl_module):
        # Access the logger path (ej. lightning_logs/version_0)
        if trainer.logger:
            version_dir = trainer.logger.version
            if isinstance(version_dir, int):
                version_dir = f"version_{version_dir}"
            
            # Build the dynamic path
            self.save_dir = os.path.join(self.base_save_dir, version_dir)
        else:
            self.save_dir = self.base_save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve metrics from the trainer
        metrics = trainer.callback_metrics
        
        t_loss = metrics.get('train_loss')
        v_loss = metrics.get('val_loss')
        t_dice = metrics.get('train_dice')
        v_dice = metrics.get('val_dice')
        g_cuantica = metrics.get('gamma_weight')

        # Only add if the value exists in this epoch
        if t_loss is not None:
            self.history['train_loss'].append(t_loss.item())
        if v_loss is not None:
            self.history['val_loss'].append(v_loss.item())
        if t_dice is not None:
            self.history['train_dice'].append(t_dice.item())
        if v_dice is not None:
            self.history['val_dice'].append(v_dice.item())
        if g_cuantica is not None:
            self.history['gamma_cuantica'].append(g_cuantica.item())

        # Generate the cumulative plot
        self._generate_loss_plot(trainer.current_epoch)
        self._generate_dice_plot(trainer.current_epoch)
        self._generate_quantum_gamma_plot(trainer.current_epoch)

    def _generate_loss_plot(self, epoch):
        plt.figure(figsize=(10, 6))
        
        epochs_range = range(len(self.history['train_loss']))

        if self.history['train_loss']:
            plt.plot(epochs_range, self.history['train_loss'], label='Train Loss')
        
        if self.history['val_loss']:
            # Adjust the range if val_loss has fewer data points (e.g. if not validating every epoch)
            val_range = range(len(self.history['val_loss']))
            plt.plot(val_range, self.history['val_loss'], label='Val Loss')

        plt.title(f'Loss del modelo por época')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida (Loss)')
        plt.legend()
        plt.grid(True, alpha=0.7)

        # Save the image
        plt.savefig(os.path.join(self.save_dir, f"loss_grafica.png"))
        plt.close()

    def _generate_dice_plot(self, epoch):
        plt.figure(figsize=(10, 6))
        
        epochs_range = range(len(self.history['train_dice']))

        if self.history['train_dice']:
            plt.plot(epochs_range, self.history['train_dice'], label='Train Dice')
        
        if self.history['val_dice']:
            # Adjust the range if val_loss has fewer data points (e.g. if not validating every epoch)
            val_range = range(len(self.history['val_dice']))
            plt.plot(val_range, self.history['val_dice'], label='Val Dice')

        plt.title(f'Coeficiente de Dice por época')
        plt.xlabel('Épocas')
        plt.ylabel('Coeficiente de Dice')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.7)

        # Save the image
        plt.savefig(os.path.join(self.save_dir, f"dice_grafica.png"))
        plt.close()

    def _generate_quantum_gamma_plot(self, epoch, title='Porcentaje de la capa cuántica utilizada', exit_file='gamma_cuantica.png'):
        plt.figure(figsize=(10, 6))
        
        epochs_range = range(len(self.history['gamma_cuantica']))

        if self.history['gamma_cuantica']:
            plt.plot(epochs_range, self.history['gamma_cuantica'], label='Gamma Cuántica')

        plt.title('Porcentaje de la capa cuántica utilizada')
        plt.xlabel('Épocas')
        plt.ylabel('Gamma cuántica')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.7)

        # Save the image
        plt.savefig(os.path.join(self.save_dir, 'gamma_cuantica.png'))
        plt.close()
