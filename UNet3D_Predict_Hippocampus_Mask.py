import streamlit as st
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tempfile


# 3D UNet Architecture
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()

        self.pool = nn.MaxPool3d(2)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoders.append(self.double_conv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        # Decoder
        for feature in reversed(features):
            self.decoders.append(
                nn.ConvTranspose3d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.decoders.append(self.double_conv(feature*2, feature))

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for enc in self.encoders:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.decoders), 2):
            x = self.decoders[idx](x)
            skip = skip_connections[idx//2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoders[idx+1](x)

        return self.final_conv(x)


# Utility Functions
def normalize(img):
    return (img - img.mean()) / (img.std() + 1e-8)


@st.cache_resource
def load_model(model_path, device):
    model = UNet3D().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_volume_3d(model, img_data, device):
    img_norm = normalize(img_data)

    # Shape: [1,1,D,H,W]
    inp = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(inp)
        pred = torch.sigmoid(pred).cpu().numpy()[0,0]

    pred_mask = (pred > 0.5).astype(np.uint8)
    return pred_mask


def save_mask(mask, affine):
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "predicted_mask_3d.nii")

    mask = mask.astype(np.uint8)
    nifti_img = nib.Nifti1Image(mask, affine)
    nib.save(nifti_img, tmp_path)

    return tmp_path


# Streamlit UI
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
st.markdown("**<span style='font-size:25px'>Upload a 3D NIfTI MRI Image</span>**", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="nii")

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    img = nib.load(tmp_file_path)
    img_data = img.get_fdata()
    affine = img.affine

    st.write(f"Uploaded MRI Shape: {img_data.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = st.text_input(
        "Path to trained 3D U-Net Model",
        value="unet3d.pth"
    )

    if os.path.exists(model_path):

        model = load_model(model_path, device)
        st.success("3D Model loaded successfully!")

        with st.spinner("Running 3D inference..."):
            mask = infer_volume_3d(model, img_data, device)

        st.success("Segmentation completed!")

        # Display middle slice
        slice_idx = img_data.shape[0] // 2
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(img_data[slice_idx], cmap="gray")
        ax.imshow(mask[slice_idx], alpha=0.4, cmap="Reds")
        ax.axis("off")

        st.pyplot(fig)

        # Download
        file_path = save_mask(mask, affine)

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        st.download_button(
            "Download Predicted 3D Mask",
            data=file_bytes,
            file_name="predicted_mask_3d.nii",
            mime="application/octet-stream"
        )

    else:
        st.warning("Model path does not exist. Please provide the correct path.")