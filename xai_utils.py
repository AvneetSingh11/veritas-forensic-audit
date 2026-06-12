import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class ScannerWrapper(torch.nn.Module):
    def __init__(self, scanner):
        super().__init__()
        self.scanner = scanner

    def forward(self, x):
        # x is a single frame tensor: (Batch, 3, 224, 224)
        tex_feats = self.scanner.texture_backbone(x)
        glob_feats = self.scanner.global_backbone(x)
        
        combined_vis = torch.cat([tex_feats, glob_feats], dim=1)
        vis_embeddings = self.scanner.vis_project(combined_vis).unsqueeze(1)
        
        audio_tensor = torch.zeros(x.size(0), 1, 64, 200).to(x.device)
        raw_audio_feats = self.scanner.audio_backbone(audio_tensor)
        audio_embeddings = self.scanner.audio_project(raw_audio_feats).unsqueeze(1)
        
        attn_out, _ = self.scanner.saff_cross_attention(vis_embeddings, audio_embeddings, audio_embeddings)
        fused_features = self.scanner.layer_norm(attn_out + vis_embeddings).squeeze(1)
        
        # Return raw logit (pre-sigmoid) to prevent vanishing gradients in Grad-CAM
        logit = self.scanner.tier1_2_classifier(fused_features)
        return logit

def generate_forensic_heatmap(model, input_tensor, original_image_np):
    """
    model: Your trained ProductionEnterpriseScanner
    input_tensor: The (1, 3, 224, 224) video frame tensor
    original_image_np: The original RGB image scaled between [0, 1] for overlay
    """
    # 1. Temporarily unfreeze the backbone so Grad-CAM can track the visual gradients
    for param in model.texture_backbone.parameters():
        param.requires_grad = True

    # 2. Wrap the model to ensure Grad-CAM tracks the entire computational graph 
    # to the final "FAKE" confidence logit.
    wrapper = ScannerWrapper(model)
    target_layers = [model.texture_backbone.features[-1]]
    
    # 3. Initialize Grad-CAM
    cam = GradCAM(model=wrapper, target_layers=target_layers)
    
    # Target the single output column (index 0) which is our FAKE probability
    targets = [ClassifierOutputTarget(0)]
    
    # 4. Generate the raw grayscale heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    # 5. Overlay the heatmap onto the original image
    visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
    
    return visualization
