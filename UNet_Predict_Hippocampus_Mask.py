import streamlit as st
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile

# U-Net Architecture
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder blocks
        for feature in features:
            self.encoders.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        # Decoder blocks
        for feature in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoders.append(self.double_conv(feature*2, feature))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)  # ConvTranspose (upsample)
            skip_conn = skip_connections[idx//2]
            # In case of mismatched sizes due to pooling
            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])
            x = torch.cat((skip_conn, x), dim=1)  # concatenate skip connection
            x = self.decoders[idx+1](x)  # double conv

        return self.final_conv(x)
    
# Utility Functions
def normalize(img):
    return (img - img.mean()) / (img.std() + 1e-8)

@st.cache_data
def load_model(model_path, device):
    model = UNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def infer_volume(model, img_data, device):
    img_norm = normalize(img_data)
    pred_mask = np.zeros_like(img_data)

    with torch.no_grad():
        for z in range(img_data.shape[0]):
            slice_img = img_norm[z]
            inp = torch.tensor(slice_img).unsqueeze(0).unsqueeze(0).float().to(device)
            pred = model(inp)
            pred = torch.sigmoid(pred).cpu().numpy()[0,0]
            pred_mask[z] = (pred > 0.5).astype(np.uint8)
    return pred_mask

def save_mask(mask, affine):
    """Save NIfTI to in-memory BytesIO for download"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".nii")
    nib.save(nib.Nifti1Image(mask, affine), tmp_file.name)
    tmp_file.close()
    return tmp_file.name

# Streamlit App UI
st.title("Brain MRI Hippocampus Segmentation")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #B0E0E6;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Upload NIfTI MRI
st.markdown("**<span style='font-size:25px'>Upload a 3D NIfTI MRI (.nii)</span>**", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="nii")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    img = nib.load(tmp_file_path)
    img_data = img.get_fdata()
    affine = img.affine
    st.write(f"Uploaded MRI shape: {img_data.shape}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load trained model
    model_path = st.text_input("Path to trained U-Net model (.pth)", value=r"C:\Users\chaudadi\Projects\unet2d.pth")
    if os.path.exists(model_path):
        model = load_model(model_path, device)
        st.success("Model loaded successfully!")

        # Predict mask
        with st.spinner("Predicting hippocampus mask..."):
            mask = infer_volume(model, img_data, device)
        st.success("Prediction done!")

        # Display middle slice overlay
        slice_idx = img_data.shape[0] // 2
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(img_data[slice_idx], cmap="gray")
        ax.imshow(mask[slice_idx], alpha=0.4, cmap="Reds")
        ax.axis("off")
        st.pyplot(fig)

        # Download predicted mask
        buffer = save_mask(mask, affine)
        st.download_button("Download Predicted Mask", buffer, file_name="predicted_mask.nii", mime="application/octet-stream")
    else:
        st.warning("Model path does not exist. Please provide correct path.")