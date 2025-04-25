import torch
from thop import profile

from engine.backbone.hgnetv2 import HGNetv2


def calculate_gflops(model, input_size=(3, 644, 644)):
    input_tensor = torch.randn(
        1,
        *input_size,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    flops, params = profile(model, inputs=(input_tensor,))
    return flops / 1e9  # Convert to GFLOPs


def main():
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = (3, 644, 644)

    # HGNetv2 models
    hgnetv2_b0 = HGNetv2("B0", pretrained=False).to(device)
    hgnetv2_b1 = HGNetv2("B1", pretrained=False).to(device)
    hgnetv2_b2 = HGNetv2("B2", pretrained=False).to(device)
    hgnetv2_b3 = HGNetv2("B3", pretrained=False).to(device)
    hgnetv2_b4 = HGNetv2("B4", pretrained=False).to(device)
    hgnetv2_b5 = HGNetv2("B5", pretrained=False).to(device)

    # DINOv2 with registers
    dinov2_vits14_reg = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14_reg"
    ).to(device)
    dinov2_vitb14_reg = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14_reg"
    ).to(device)
    dinov2_vitl14_reg = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_reg"
    ).to(device)
    dinov2_vitg14_reg = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitg14_reg"
    ).to(device)

    # Create lists to store results
    results = []

    # Calculate GFLOPs and VRAM usage for HGNetv2 models
    for model, name in [
        (hgnetv2_b0, "HGNetV2-B0"),
        (hgnetv2_b1, "HGNetV2-B1"),
        (hgnetv2_b2, "HGNetV2-B2"),
        (hgnetv2_b3, "HGNetV2-B3"),
        (hgnetv2_b4, "HGNetV2-B4"),
        (hgnetv2_b5, "HGNetV2-B5"),
    ]:
        model.eval().cuda() if torch.cuda.is_available() else model.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Measure FLOPs (assuming your function works as expected)
            gflops = calculate_gflops(model)

            # Get peak memory usage
            vram_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

        results.append((name, gflops, vram_mb))
        torch.cuda.empty_cache()

    # Calculate GFLOPs and VRAM usage for DINOv2 models
    for model, name in [
        (dinov2_vits14_reg, "ViT-S"),
        (dinov2_vitb14_reg, "ViT-B"),
        (dinov2_vitl14_reg, "ViT-L"),
        (dinov2_vitg14_reg, "ViT-g"),
    ]:
        model.eval().cuda() if torch.cuda.is_available() else model.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Measure FLOPs (assuming your function works as expected)
            gflops = calculate_gflops(model)

            # Get peak memory usage
            vram_mb = (
                torch.cuda.max_memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0
            )

        results.append((name, gflops, vram_mb))
        torch.cuda.empty_cache()

    # Print results in a table format
    print(f"\nModel Resource Analysis: ({input_size[1:]} input size)")
    print("=" * 50)
    print(f"{'Model':<20} {'GFLOPs':>10} {'VRAM (MB)':>12}")
    print("-" * 50)
    for name, gflops, vram in results:
        print(f"{name:<20} {gflops:>10.2f} {vram:>12.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
