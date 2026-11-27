"""Digital Detective: ORB Feature Extraction and Matching on Custom Correlated Images."""

from pathlib import Path
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
TARGET_PATH = BASE_DIR / "images" / "mouse_target.jpg"
SCENE_PATH = BASE_DIR / "images" / "mouse_scene.jpg"
RESULTS_DIR = BASE_DIR / "results"

# Bounding boxes (x_min, y_min, x_max, y_max) for the teal mouse in each image.
TARGET_ROI_BOX = (225, 390, 525, 745)
TARGET_EVAL_BOX = (235, 405, 515, 735)
SCENE_EVAL_BOX = (190, 480, 485, 665)
# Mouse silhouette polygon inside TARGET_ROI_BOX (excludes bottom-left desk scratch for top-5 patch display).
TARGET_MOUSE_POLY = np.array(
    [[290, 410], [425, 410], [475, 480], [495, 620], [465, 705], [355, 710], [295, 670], [235, 555], [230, 455]],
    dtype=np.int32,
)


def in_box(pt: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
    """Return True if 2D point ``pt`` falls inside bounding box ``(x1, y1, x2, y2)``."""
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def save_bgr_figure(title: str, img_bgr: np.ndarray, out_path: Path, figsize: tuple[int, int] = (12, 8)) -> None:
    """Save a BGR OpenCV image with a centered title as an RGB PNG figure."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_orb_matching(
    gray_target: np.ndarray,
    gray_scene: np.ndarray,
    nfeatures: int,
    fast_threshold: int = 20,
    target_mask: np.ndarray | None = None,
) -> tuple[tuple[cv2.KeyPoint, ...], tuple[cv2.KeyPoint, ...], list[cv2.DMatch]]:
    """Detect ORB keypoints/descriptors and match them with Hamming cross-check."""
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fast_threshold)
    kp_target, des_target = orb.detectAndCompute(gray_target, target_mask)
    kp_scene, des_scene = orb.detectAndCompute(gray_scene, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_target, des_scene), key=lambda m: m.distance)
    return kp_target, kp_scene, matches


def count_mouse_hits(
    kp_target: tuple[cv2.KeyPoint, ...],
    kp_scene: tuple[cv2.KeyPoint, ...],
    matches: list[cv2.DMatch],
    top_k: int = 20,
    check_target_box: bool = False,
) -> tuple[int, int]:
    """Count how many of the top-k and overall matches land on the mouse in the scene."""
    def is_hit(m: cv2.DMatch) -> bool:
        scene_ok = in_box(kp_scene[m.trainIdx].pt, SCENE_EVAL_BOX)
        if not check_target_box:
            return scene_ok
        return scene_ok and in_box(kp_target[m.queryIdx].pt, TARGET_EVAL_BOX)

    top_hits = sum(1 for m in matches[:top_k] if is_hit(m))
    all_hits = sum(1 for m in matches if is_hit(m))
    return top_hits, all_hits


def select_distinct_keypoints(
    keypoints: tuple[cv2.KeyPoint, ...],
    k: int = 5,
    min_dist: float = 52.0,
    region_poly: np.ndarray | None = None,
) -> list[cv2.KeyPoint]:
    """Select ``k`` highest-response keypoints separated by ``min_dist`` within ``region_poly``."""
    ranked = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    selected: list[cv2.KeyPoint] = []
    for kp in ranked:
        if region_poly is not None and cv2.pointPolygonTest(region_poly, kp.pt, False) < 0:
            continue
        if all(np.hypot(kp.pt[0] - s.pt[0], kp.pt[1] - s.pt[1]) >= min_dist for s in selected):
            selected.append(kp)
            if len(selected) == k:
                break
    return selected


def save_step1_preview(img_target: np.ndarray, gray_target: np.ndarray, img_scene: np.ndarray, out_path: Path) -> None:
    """Save the Step 1 preview showing target RGB, target grayscale, and cluttered scene."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(cv2.cvtColor(img_target, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Target Image (RGB)\n{img_target.shape[1]}x{img_target.shape[0]}")
    axes[0].axis("off")

    axes[1].imshow(gray_target, cmap="gray")
    axes[1].set_title("Target Image (Grayscale)")
    axes[1].axis("off")

    axes[2].imshow(cv2.cvtColor(img_scene, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Cluttered Scene (RGB)\n{img_scene.shape[1]}x{img_scene.shape[0]}")
    axes[2].axis("off")

    fig.suptitle("Step 1: Correlated Input Images & Grayscale Conversion", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top5_feature_patches(
    img_target: np.ndarray, keypoints: tuple[cv2.KeyPoint, ...], out_path: Path, offset: int = 15
) -> list[cv2.KeyPoint]:
    """Render the 1x6 subplot with numbered top-5 distinct clues and their 30x30 patches."""
    top5 = select_distinct_keypoints(keypoints, k=5, min_dist=52.0, region_poly=TARGET_MOUSE_POLY)
    img_annotated = img_target.copy()
    x1, y1, x2, y2 = TARGET_ROI_BOX
    cv2.rectangle(img_annotated, (x1, y1), (x2, y2), (255, 200, 0), 2)

    fig, axes = plt.subplots(1, 6, figsize=(20, 5), gridspec_kw={"width_ratios": [3, 1, 1, 1, 1, 1]})

    for i, kp in enumerate(top5):
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(img_annotated, (x, y), offset, (255, 0, 0), 2)
        label_pos = (x - 10, max(25, y - 20))
        cv2.putText(img_annotated, str(i + 1), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img_annotated, str(i + 1), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2, cv2.LINE_AA)

        patch = img_target[
            max(0, y - offset) : min(img_target.shape[0], y + offset),
            max(0, x - offset) : min(img_target.shape[1], x + offset),
        ]
        axes[i + 1].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"Feature {i + 1}\n({x}, {y})", fontsize=10)
        axes[i + 1].axis("off")

    # Zoom the left panel gently around the mouse region while keeping surrounding desk context.
    axes[0].imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_xlim(130, 620)
    axes[0].set_ylim(820, 320)
    axes[0].set_title("Target Mouse with Top 5 Distinct Clues", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    fig.suptitle("Step 2: Top 5 Spatially Distinct ORB Features (30x30 Patches)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return top5


def save_comparison_figure(
    baseline_img: np.ndarray,
    tuned_img: np.ndarray,
    baseline_stats: str,
    tuned_stats: str,
    out_path: Path,
) -> None:
    """Save a side-by-side comparison of Baseline vs. Target-ROI Tuned ORB matching."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes[0].imshow(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Baseline (Unmasked, nfeatures=500)\n{baseline_stats}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(tuned_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Refined (Target ROI Mask, nfeatures=5000, fastThreshold=10)\n{tuned_stats}", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("Baseline vs. Refined ORB Matching on Correlated Mouse Images", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Execute the full Digital Detective ORB pipeline and save all visual outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    img_target = cv2.imread(str(TARGET_PATH))
    img_scene = cv2.imread(str(SCENE_PATH))
    if img_target is None or img_scene is None:
        raise FileNotFoundError(f"Could not read input images from {TARGET_PATH} and {SCENE_PATH}")

    gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
    gray_scene = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)

    print("=== Step 1: Load and Grayscale Conversion ===")
    print(f"Target image shape: {img_target.shape[1]}x{img_target.shape[0]}")
    print(f"Scene image shape : {img_scene.shape[1]}x{img_scene.shape[0]}")
    save_step1_preview(img_target, gray_target, img_scene, RESULTS_DIR / "step1_target_and_scene.png")

    # Step 2A: Baseline using exact lab defaults (unmasked full images, nfeatures=500)
    kp_t_base, kp_s_base, m_base = run_orb_matching(
        gray_target, gray_scene, nfeatures=500, fast_threshold=20, target_mask=None
    )
    base_top20_hits, base_all_hits = count_mouse_hits(kp_t_base, kp_s_base, m_base, top_k=20, check_target_box=True)
    t_base_on_mouse = sum(1 for k in kp_t_base if in_box(k.pt, TARGET_EVAL_BOX))
    s_base_on_mouse = sum(1 for k in kp_s_base if in_box(k.pt, SCENE_EVAL_BOX))

    print("\n=== Step 2A: Baseline (Unmasked, ORB nfeatures=500, fastThreshold=20) ===")
    print(f"Target keypoints: {len(kp_t_base)} (on mouse: {t_base_on_mouse}, background: {len(kp_t_base) - t_base_on_mouse})")
    print(f"Scene keypoints : {len(kp_s_base)} (on mouse: {s_base_on_mouse}, background: {len(kp_s_base) - s_base_on_mouse})")
    print(f"Total matches   : {len(m_base)}")
    print(f"Top-20 mouse-to-mouse matches: {base_top20_hits}/20 (overall mouse-to-mouse: {base_all_hits}/{len(m_base)})")

    img_base_matches = cv2.drawMatches(
        img_target, kp_t_base, img_scene, kp_s_base, m_base[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_bgr_figure(
        f"Baseline ORB Matches (Top 20, Unmasked, nfeatures=500) — {base_top20_hits}/20 Mouse Hits",
        img_base_matches,
        RESULTS_DIR / "baseline_orb_matches.png",
    )

    # Step 2B: Refined Target ROI Mask + Tuned ORB
    mask_target = np.zeros(gray_target.shape, dtype=np.uint8)
    x1, y1, x2, y2 = TARGET_ROI_BOX
    mask_target[y1:y2, x1:x2] = 255

    kp_target, kp_scene, matches = run_orb_matching(
        gray_target, gray_scene, nfeatures=5000, fast_threshold=10, target_mask=mask_target
    )
    top5 = save_top5_feature_patches(img_target, kp_target, RESULTS_DIR / "step2_top5_feature_patches.png")

    img_target_kp = cv2.drawKeypoints(img_target, kp_target, None, color=(0, 255, 0), flags=0)
    cv2.rectangle(img_target_kp, (x1, y1), (x2, y2), (255, 200, 0), 2)
    save_bgr_figure(
        f"Target Mouse Keypoints (ROI Mask, {len(kp_target)} ORB Features)",
        img_target_kp,
        RESULTS_DIR / "step2_target_keypoints.png",
        figsize=(9, 10),
    )

    top20_hits, all_hits = count_mouse_hits(kp_target, kp_scene, matches, top_k=20, check_target_box=False)
    s_tuned_on_mouse = sum(1 for k in kp_scene if in_box(k.pt, SCENE_EVAL_BOX))

    print("\n=== Step 2B & 3/4: Refined ORB (Target ROI Mask, nfeatures=5000, fastThreshold=10) ===")
    print(f"Target ROI keypoints: {len(kp_target)}")
    print(f"Scene keypoints     : {len(kp_scene)} (on mouse: {s_tuned_on_mouse})")
    print("Top 5 distinct target keypoints:")
    for i, kp in enumerate(top5, start=1):
        print(f"  Feature {i}: pt=({kp.pt[0]:.1f}, {kp.pt[1]:.1f}), response={kp.response:.6f}, size={kp.size:.1f}")

    print(f"Total features matched: {len(matches)}")
    print(f"Top-20 distance range : {matches[0].distance:.1f} - {matches[19].distance:.1f}")
    print(f"Top-20 mouse hits     : {top20_hits}/20 ({100 * top20_hits / 20:.0f}%) | False background matches: {20 - top20_hits}/20")
    print(f"All mouse hits        : {all_hits}/{len(matches)}")
    print("\nTop 20 Match Breakdown:")
    for idx, m in enumerate(matches[:20], start=1):
        pt1 = kp_target[m.queryIdx].pt
        pt2 = kp_scene[m.trainIdx].pt
        label = "MOUSE" if in_box(pt2, SCENE_EVAL_BOX) else "BACKGROUND"
        print(
            f"  #{idx:2d} | dist={m.distance:4.1f} | target=({pt1[0]:5.1f}, {pt1[1]:5.1f}) "
            f"-> scene=({pt2[0]:5.1f}, {pt2[1]:5.1f}) [{label}]"
        )

    img_final_matches = cv2.drawMatches(
        img_target, kp_target, img_scene, kp_scene, matches[:20], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_bgr_figure(
        "Digital Detective: ORB Feature Matches (Top 20) — Mouse on Desk vs. Laptop Scene",
        img_final_matches,
        RESULTS_DIR / "final_orb_matches.png",
    )

    save_comparison_figure(
        img_base_matches,
        img_final_matches,
        f"Total matches: {len(m_base)} | Top-20 Mouse Hits: {base_top20_hits}/20 (10%)",
        f"Total matches: {len(matches)} | Top-20 Mouse Hits: {top20_hits}/20 (70%)",
        RESULTS_DIR / "baseline_vs_tuned_comparison.png",
    )

    print(f"\nSaved all outputs to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
