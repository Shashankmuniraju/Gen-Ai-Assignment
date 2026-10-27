"""
VAE Latent Space Visualizer
Interactive Streamlit application for encoding, visualizing, and decoding MNIST in latent space.
"""

import os
import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Import backend modules
from src.model import VAE
from src.dataset import get_mnist_dataloaders, get_sample_digits
from src.inference import (
    load_model,
    preprocess_image,
    reconstruct_and_diagnose,
    generate_latent_grid,
    compute_latent_dataset_embeddings,
    interpolate_latent,
)

# Page configuration
st.set_page_config(
    page_title="VAE Latent Space Visualizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern visual design
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_model(checkpoint_path: str = "models/vae_mnist.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path=checkpoint_path, latent_dim=2, device=device)
    return model, device


@st.cache_resource
def get_cached_dataset():
    _, test_loader = get_mnist_dataloaders(data_dir="./data", batch_size=128)
    sample_dict = get_sample_digits(test_loader, samples_per_digit=5)
    return test_loader, sample_dict


@st.cache_data
def get_cached_embeddings(_model, _test_loader, max_samples=1500):
    return compute_latent_dataset_embeddings(_model, _test_loader, max_samples=max_samples)


def main():
    st.title("🧬 Variational Autoencoder (VAE) Latent Space Visualizer")
    st.caption("Deep Generative Modeling on MNIST — PyTorch & Streamlit")

    # Load resources
    checkpoint_path = "models/vae_mnist.pth"
    model, device = get_cached_model(checkpoint_path)
    test_loader, sample_dict = get_cached_dataset()

    # Sidebar Controls
    st.sidebar.header("⚙️ Controls & Configuration")

    st.sidebar.subheader("1. Input Image Source")
    input_mode = st.sidebar.radio(
        "Choose Input Method",
        ["Select Sample Digit", "Upload Custom Image", "Random Test Sample"],
        index=0,
    )

    image_tensor = None
    input_caption = ""

    if input_mode == "Select Sample Digit":
        col_d, col_v = st.sidebar.columns(2)
        with col_d:
            selected_digit = st.selectbox("Digit (0-9)", list(range(10)), index=3)
        with col_v:
            max_var = len(sample_dict.get(selected_digit, []))
            selected_var = st.selectbox("Sample Index", list(range(max_var)), index=0)

        if selected_digit in sample_dict and len(sample_dict[selected_digit]) > selected_var:
            image_tensor = sample_dict[selected_digit][selected_var].unsqueeze(0)
            input_caption = f"MNIST Digit '{selected_digit}' (Sample #{selected_var})"

    elif input_mode == "Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader(
            "Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            image_tensor = preprocess_image(uploaded_file)
            input_caption = f"Uploaded: {uploaded_file.name}"
        else:
            st.sidebar.info("Upload an image or fall back to sample digit.")
            image_tensor = sample_dict[0][0].unsqueeze(0)
            input_caption = "Default: Digit '0'"

    elif input_mode == "Random Test Sample":
        if "random_seed" not in st.session_state:
            st.session_state.random_seed = 42

        if st.sidebar.button("🎲 Draw Another Random Sample"):
            st.session_state.random_seed = np.random.randint(0, 10000)

        rng = np.random.RandomState(st.session_state.random_seed)
        rand_digit = rng.choice(list(sample_dict.keys()))
        rand_idx = rng.choice(len(sample_dict[rand_digit]))
        image_tensor = sample_dict[rand_digit][rand_idx].unsqueeze(0)
        input_caption = f"Random Sample: Digit '{rand_digit}'"

    st.sidebar.divider()
    st.sidebar.subheader("2. Latent & Loss Hyperparameters")
    beta = st.sidebar.slider(
        "KL Divergence Weight (β-VAE)",
        min_value=0.0,
        max_value=4.0,
        value=1.0,
        step=0.1,
        help="Controls the trade-off between reconstruction fidelity and prior regularization.",
    )
    sampling_mode = st.sidebar.radio(
        "Latent Sampling Mode",
        ["Deterministic (z = μ)", "Stochastic (z ~ N(μ, σ²))"],
        index=0,
    )
    deterministic = sampling_mode.startswith("Deterministic")

    # Run inference for the selected image
    diagnostics = reconstruct_and_diagnose(
        model=model,
        image_tensor=image_tensor,
        device=device,
        beta=beta,
        deterministic=deterministic,
    )

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Encode & Reconstruct",
        "🗺️ 2D Latent Space Map",
        "🎛️ Manifold Grid & Explorer",
        "🔄 Latent Morphing / Interpolation",
        "📐 Model Diagnostics & Math",
    ])

    # =========================================================================
    # TAB 1: ENCODE & RECONSTRUCT
    # =========================================================================
    with tab1:
        st.subheader("Image Reconstruction & Error Diagnostic")
        st.write(
            f"Active Input: **{input_caption}** | Latent Dimension = **2** | Device = `{device}`"
        )

        col_orig, col_recon, col_diff = st.columns(3)

        with col_orig:
            st.markdown("**Original Input Image** ($x$)")
            fig_orig, ax_orig = plt.subplots(figsize=(3.5, 3.5))
            ax_orig.imshow(diagnostics["original_np"], cmap="gray", vmin=0, vmax=1)
            ax_orig.axis("off")
            st.pyplot(fig_orig, use_container_width=True)
            plt.close(fig_orig)

        with col_recon:
            st.markdown(r"**Reconstructed Image** ($\hat{x}$)")
            fig_recon, ax_recon = plt.subplots(figsize=(3.5, 3.5))
            ax_recon.imshow(diagnostics["reconstructed_np"], cmap="gray", vmin=0, vmax=1)
            ax_recon.axis("off")
            st.pyplot(fig_recon, use_container_width=True)
            plt.close(fig_recon)

        with col_diff:
            st.markdown(r"**Absolute Difference Heatmap** ($|x - \hat{x}|$)")
            fig_diff, ax_diff = plt.subplots(figsize=(3.5, 3.5))
            im_diff = ax_diff.imshow(diagnostics["diff_np"], cmap="magma", vmin=0, vmax=1)
            ax_diff.axis("off")
            fig_diff.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
            st.pyplot(fig_diff, use_container_width=True)
            plt.close(fig_diff)

        # Quantitative Metrics Row
        st.markdown("### 📊 Loss Components & Diagnostics")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Reconstruction Loss (BCE)</div>
                    <div class="metric-value">{diagnostics['recon_loss_bce']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Squared Error (MSE)</div>
                    <div class="metric-value">{diagnostics['recon_loss_mse']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">KL Divergence (D_KL)</div>
                    <div class="metric-value">{diagnostics['kl_divergence']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Total ELBO Loss (BCE + β·KL)</div>
                    <div class="metric-value">{diagnostics['total_loss']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("🧬 Latent Vector Representation")
        z_coords = diagnostics["z"]
        mu_coords = diagnostics["mu"]
        std_coords = diagnostics["std"]

        col_z1, col_z2 = st.columns(2)
        with col_z1:
            st.info(
                f"**Latent Mean $\\boldsymbol{{\\mu}}$:** `[{mu_coords[0]:+.4f}, {mu_coords[1]:+.4f}]`\n\n"
                f"**Latent Standard Deviation $\\boldsymbol{{\\sigma}}$:** `[{std_coords[0]:.4f}, {std_coords[1]:.4f}]`"
            )
        with col_z2:
            st.success(
                f"**Sampled Latent Vector $\\mathbf{{z}}$:** `[{z_coords[0]:+.4f}, {z_coords[1]:+.4f}]`\n\n"
                f"**Sampling Mode:** `{sampling_mode}`"
            )

    # =========================================================================
    # TAB 2: 2D LATENT SPACE MAP
    # =========================================================================
    with tab2:
        st.subheader("2D Latent Space Projection & Active Sample Position")
        st.write(
            r"Each point represents an encoded MNIST test set image colored by its ground-truth digit. "
            r"The **red crosshair star (★)** denotes the active input's latent coordinate $(\mu_1, \mu_2)$."
        )

        df_embeddings = get_cached_embeddings(model, test_loader, max_samples=1200)

        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        # Custom palette for 10 digits
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for d in range(10):
            sub = df_embeddings[df_embeddings["digit"] == str(d)]
            ax_scatter.scatter(
                sub["z1"],
                sub["z2"],
                c=[colors[d]],
                label=f"Digit {d}",
                alpha=0.6,
                s=20,
                edgecolors="none",
            )

        # Highlight current point
        curr_mu = diagnostics["mu"]
        ax_scatter.scatter(
            curr_mu[0],
            curr_mu[1],
            color="red",
            s=220,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            label=f"Current: ({curr_mu[0]:.2f}, {curr_mu[1]:.2f})",
            zorder=10,
        )

        ax_scatter.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax_scatter.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax_scatter.set_xlabel("Latent Dimension $z_1$", fontsize=11)
        ax_scatter.set_ylabel("Latent Dimension $z_2$", fontsize=11)
        ax_scatter.set_title("Learned 2D Latent Space Manifold", fontsize=13, fontweight="bold")
        ax_scatter.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        ax_scatter.grid(True, alpha=0.2)
        fig_scatter.tight_layout()

        st.pyplot(fig_scatter, use_container_width=True)
        plt.close(fig_scatter)

    # =========================================================================
    # TAB 3: MANIFOLD GRID & DIRECT EXPLORER
    # =========================================================================
    with tab3:
        st.subheader("Interactive Latent Navigation & Continuous Manifold")

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            st.markdown("#### 🎮 Direct Latent Vector Synthesis")
            st.caption("Manually adjust latent coordinates to synthesize novel digits in real-time:")
            user_z1 = st.slider("Coordinate $z_1$", min_value=-3.0, max_value=3.0, value=float(curr_mu[0]), step=0.1)
            user_z2 = st.slider("Coordinate $z_2$", min_value=-3.0, max_value=3.0, value=float(curr_mu[1]), step=0.1)

            # Synthesize digit from custom coordinates
            with torch.no_grad():
                z_custom = torch.tensor([[user_z1, user_z2]], dtype=torch.float32, device=device)
                synth_digit = model.decode(z_custom).squeeze().cpu().numpy()

            fig_synth, ax_synth = plt.subplots(figsize=(3, 3))
            ax_synth.imshow(synth_digit, cmap="gray", vmin=0, vmax=1)
            ax_synth.axis("off")
            ax_synth.set_title(f"Decoded from z=[{user_z1:.2f}, {user_z2:.2f}]", fontsize=10)
            st.pyplot(fig_synth, use_container_width=False)
            plt.close(fig_synth)

        with col_ctrl2:
            st.markdown("#### 🗺️ 2D Latent Manifold Meshgrid")
            st.caption("Decodes a regular coordinate grid across the prior space:")
            grid_res = st.slider("Grid Resolution (NxN)", min_value=10, max_value=25, value=18, step=1)
            grid_bound = st.slider("Coordinate Bound [-B, +B]", min_value=1.5, max_value=3.5, value=2.5, step=0.25)

            grid_img = generate_latent_grid(model, grid_size=grid_res, bound=grid_bound, device=device)

            fig_grid, ax_grid = plt.subplots(figsize=(6, 6))
            ax_grid.imshow(grid_img, cmap="gnuplot2", vmin=0, vmax=1)
            ax_grid.axis("off")
            ax_grid.set_title(f"Continuous Latent Manifold ({grid_res}x{grid_res})", fontsize=11)
            st.pyplot(fig_grid, use_container_width=True)
            plt.close(fig_grid)

    # =========================================================================
    # TAB 4: LATENT MORPHING / INTERPOLATION
    # =========================================================================
    with tab4:
        st.subheader("Latent Space Morphing & Geodesic Interpolation")
        st.write(
            "Traverse the continuous latent manifold between two digits. The VAE reconstructs "
            "smooth, realistic intermediate forms rather than cross-dissolving pixel values."
        )

        col_i1, col_i2, col_i3 = st.columns(3)
        with col_i1:
            digit_a = st.selectbox("Start Digit (A)", list(range(10)), index=1)
        with col_i2:
            digit_b = st.selectbox("End Digit (B)", list(range(10)), index=8)
        with col_i3:
            interp_steps = st.slider("Interpolation Steps", min_value=5, max_value=20, value=10, step=1)

        # Get embeddings for digit_a and digit_b
        img_a = sample_dict[digit_a][0].unsqueeze(0).to(device)
        img_b = sample_dict[digit_b][0].unsqueeze(0).to(device)

        with torch.no_grad():
            mu_a, _ = model.encode(img_a)
            mu_b, _ = model.encode(img_b)
            z_a = mu_a.squeeze().cpu().numpy()
            z_b = mu_b.squeeze().cpu().numpy()

        strip_img, frames = interpolate_latent(model, z_a, z_b, steps=interp_steps, device=device)

        fig_strip, ax_strip = plt.subplots(figsize=(10, 2))
        ax_strip.imshow(strip_img, cmap="gray", vmin=0, vmax=1)
        ax_strip.axis("off")
        ax_strip.set_title(
            f"Morphing Sequence: Digit {digit_a} (z=[{z_a[0]:.2f}, {z_a[1]:.2f}]) → Digit {digit_b} (z=[{z_b[0]:.2f}, {z_b[1]:.2f}])",
            fontsize=10,
        )
        st.pyplot(fig_strip, use_container_width=True)
        plt.close(fig_strip)

        # Step slider inspection
        step_idx = st.slider("Inspect Individual Frame", min_value=0, max_value=interp_steps - 1, value=interp_steps // 2)
        alpha = step_idx / (interp_steps - 1)
        col_frame, col_info = st.columns([1, 3])
        with col_frame:
            fig_frame, ax_frame = plt.subplots(figsize=(2.5, 2.5))
            ax_frame.imshow(frames[step_idx], cmap="gray", vmin=0, vmax=1)
            ax_frame.axis("off")
            st.pyplot(fig_frame, use_container_width=False)
            plt.close(fig_frame)
        with col_info:
            z_current_interp = (1.0 - alpha) * z_a + alpha * z_b
            st.markdown(
                f"""
                - **Interpolation Weight (α):** `{alpha:.2f}` (Frame {step_idx + 1} of {interp_steps})
                - **Interpolated Latent Coordinate:** `[{z_current_interp[0]:+.4f}, {z_current_interp[1]:+.4f}]`
                - **Transition Characteristic:** Demonstrates smooth topological deformations of stroke structures!
                """
            )

    # =========================================================================
    # TAB 5: ARCHITECTURE & MATHEMATICAL THEORY
    # =========================================================================
    with tab5:
        st.subheader("Variational Autoencoder (VAE) Theoretical Foundations")

        st.markdown(
            r"""
            ### 1. Evidence Lower Bound (ELBO)
            Variational Autoencoders optimize the marginal log-likelihood $\log p(x)$ via variational inference:

            $$\log p(x) \ge \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{KL}\left(q_\phi(z|x) \parallel p(z)\right)$$

            In practice, we minimize the negative ELBO:
            $$\mathcal{L}_{VAE}(\theta, \phi; x) = \mathcal{L}_{\text{Reconstruction}} + \beta \cdot D_{KL}\left(q_\phi(z|x) \parallel p(z)\right)$$

            - **Reconstruction Loss**: Binary Cross Entropy (BCE) modeling Bernoulli pixel likelihood:
              $$\mathcal{L}_{\text{Recon}} = -\sum_{i=1}^{784} \left[ x_i \log \hat{x}_i + (1 - x_i)\log(1 - \hat{x}_i) \right]$$
            - **Kullback-Leibler (KL) Divergence**: Measures distance from approximate posterior $q(z|x) = \mathcal{N}(\mu, \sigma^2)$ to standard normal prior $p(z) = \mathcal{N}(0, I)$:
              $$D_{KL} = -\frac{1}{2} \sum_{j=1}^{d} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$

            ### 2. The Reparameterization Trick
            Direct sampling $z \sim q_\phi(z|x)$ prevents gradient backpropagation. The reparameterization trick isolates stochasticity into an independent noise variable $\epsilon$:

            $$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

            $$\frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon$$

            This enables full end-to-end backpropagation through the sampling layer into the convolutional encoder.
            """
        )

        st.divider()
        st.subheader("Model Parameter Architecture")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Parameters", f"{total_params:,}")
        c2.metric("Trainable Parameters", f"{trainable_params:,}")
        c3.metric("Latent Dimensions", f"{model.latent_dim}")


if __name__ == "__main__":
    main()
