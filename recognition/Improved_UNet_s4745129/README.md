## HipMRI Segmentation with a 2D improved U-Net

# Description of the Algorithm
# Baseline UNet 
The baseline was adapted from the demo 2 UNet implementation. It uses
double convolution blocks with batch normalisation and ReLU, max pooling
for downsampling, and transposed convolutions for upsampling. Skip 
connections combine encoder features with decoder features to help 
preserve spatial detail. The model takes one-channel MRI slices and 
produces six class scores per pixel.
# Improved UNet
The Improved UNet is a 2D adaptation of the architecture described by
the source given in the reference list [2], specifically Section 2.1 
and Figure 1. The original paper uses 3D brain MRI volumes, while this
implementation uses 2D convolutions for HipMRI slices.
    The changes from the baseline include:
        Residual context blocks with dropout probability 0.3
        Stride = 2 convolutions instead of max pooling
        Instance normalisation instead of batch normalisation
        Leaky ReLU with a negative slope of 0.01
        Nearest neighbour upsampling followed by convolution instead of
            transposed convolution
        Localisation blocks that combine encoder and decoder features
        Segmentation heads at three decoder resolutions, with there
            class scores upsampled and added together before calculating
            the loss
The encoder uses 16, 32, 64, 128, 256 channels. The implementation uses
Dropout2d for channel-wise dropout. These aare documented choices for
adapting the paper to 2D.
The baseline has 31036806 parameters, while the Improved UNet has 2717186.
Since channel widths and several architectural components differ, this 
experiment compares the complete model configurations rather than measuring
the effect of any single change.
Both models passed checks for the expected output shape of [1,, 6, 256, 128].
The Improved UNet also passed a backward-pass check, with finite gradients 
for every trainable parameter. 

# Feasibility Review
1. User Need, Scope & Acceptance Criteria:
    This project will investigate a prototype that produces
    anatomical segmentation masks from the supplied HipMRI 2D slices
    for an imaging specialist to review. 
    A standard 2D U-Net will be compared with a 2D
    Improved U-Net to investigate whether the improved 
    architecture reduces boundary errors enough too justify
    its computational cost.
    The testable acceptace criteria addressing the domain's
    open dilemma will be:
        - Achieve a prostate Dice score of at least 0.75
            on the test set
        - Compare the Improved U-Net with a standard U-Net to 
            check whether it produces more accurate 
            segmentations and fewer bounadry errors
        - Examine 3-5 failure cases to identify where the 
            predicted masks would need correction by a 
            specialist
        - Run both models on one Rangpur GPU, recording
            peak GPU memory and inference time to judge
            whether any improvement is worth the extra
            resources
2. Model Choice & Course Concepts:
    The HipMRI 2D dataset will be used with the supplied patient
    separated training, validation and test splits. Furthermore,
    a normal difficulty 2D Improved U-Net will be implemented,
    and compared with an easy difficulty standard 2D U-Net 
    baseline.
    Both models use convolutions to learn image features and
    an encoder-decoder structure to produce the segmentation
    masks. Skip connectiuons help preserve spatial detail,
    which is useful for identifying anatomical boundaries.
    Additionally, the references provided in the assignment spec
    will be used to help justify the architecture and explain the
    changes made in the improved U-Net.
    Since the HipMRI dataset uses Nifti image files, Appendix B
    will be used as a guidance tool to load files.
3. Preliminary Feasibility Evidence:
    The HipMRI 2D dataset has 11460 training slices from 31 
    patients, 660 validation slices from four patients and 540 
    test slices from three patients. Each image has a corresponding
    segmentation mask with no patient overlap between the training,
    validation and test splits.
    Nine image-mask pairs were also loaded using nibabel. All
    sampled pairs had matching dimensions of 256 x 128. These
    checks provide initial evidence that data can be loaded for
    segmentation.
4. Risks, Budget & Fallback:
    Main risks include difficulty segmenting small regions and 
    boundaries, possible delays obtaining GPU resources
    and overfitting to the training patients. A small baseline
    training test will first be run on one Rangpur GPU to estimate
    training time and memory use. If this memory use is too high,
    the batch size or model size can be reduced. If the Improved
    U-Net does not perform better than the standard U-Net baseline,
    the pros and cons can be discussed and whether its extra 
    complexity is worthwhile.

# Preprocessing
Images were normalised using per-image z-score normalisation. 
Image-mask pairs with dimensions that were different to 256x128 were
resized to this size for batching. Bilinear interpolation was used for
images and nearest neighbour interpolation for masks to preserve the 
class labels. Since resizing can change proportions, the same 
preprocessing was applied to both models for a consistent comparison.

