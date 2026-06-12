import cv2
import numpy as np
import torch

def generate_forensic_heatmap(model, input_tensor, original_image_np):
    model.eval()
    
    for param in model.texture_backbone.parameters():
        param.requires_grad = True
        
    target_layer = model.texture_backbone.features[-1]
    
    activations = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
        
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
        
    hook1 = target_layer.register_forward_hook(forward_hook)
    hook2 = target_layer.register_full_backward_hook(backward_hook)
    
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
    
    hook1.remove()
    hook2.remove()
    
    if activations is None or gradients is None:
        return original_image_np
        
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze().detach().cpu().numpy()
    
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
        
    cam = cv2.resize(cam, (original_image_np.shape[1], original_image_np.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLORBGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    cam_result = heatmap * 0.5 + original_image_np * 0.5
    cam_result = cam_result / np.max(cam_result)
    
    return cam_result
