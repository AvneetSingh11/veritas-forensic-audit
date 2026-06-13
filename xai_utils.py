import cv2
import numpy as np
import torch

def generate_forensic_heatmap(model, input_tensor, original_image_np):
    model.eval()
    
    input_tensor.requires_grad_()
    
    # Forward pass through BOTH backbones
    tex_feats = model.texture_backbone(input_tensor)
    glob_feats = model.global_backbone(input_tensor)
    
    combined_vis = torch.cat([tex_feats, glob_feats], dim=1)
    vis_embeddings = model.vis_project(combined_vis).unsqueeze(1)
    
    audio_tensor = torch.zeros(input_tensor.size(0), 1, 64, 200).to(input_tensor.device)
    raw_audio_feats = model.audio_backbone(audio_tensor)
    audio_embeddings = model.audio_project(raw_audio_feats).unsqueeze(1)
    
    attn_out, _ = model.saff_cross_attention(vis_embeddings, audio_embeddings, audio_embeddings)
    fused_features = model.layer_norm(attn_out + vis_embeddings).squeeze(1)
    logit = model.tier1_2_classifier(fused_features)
    
    model.zero_grad()
    logit[0, 0].backward()
    
    # Extract gradients directly from the input pixels (incorporates both texture and structure)
    if input_tensor.grad is None:
        return original_image_np
        
    gradients = input_tensor.grad[0].cpu().numpy()
    
    # Calculate saliency by taking the max absolute gradient across color channels
    saliency = np.max(np.abs(gradients), axis=0)
    
    # Heavy Gaussian blur to simulate smooth Grad-CAM blobs
    saliency = cv2.GaussianBlur(saliency, (35, 35), 0)
    
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
        
    # Resize to original image resolution
    cam = cv2.resize(saliency, (original_image_np.shape[1], original_image_np.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    cam_result = heatmap * 0.5 + original_image_np * 0.5
    cam_result = cam_result / np.max(cam_result)
    
    # Free computational graph memory to prevent RAM leaks
    model.zero_grad(set_to_none=True)
    input_tensor.requires_grad_(False)
    
    return np.uint8(255 * cam_result)
