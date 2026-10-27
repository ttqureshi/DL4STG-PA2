# Model D reach

Model D’s one-step reach is 16 original pixels. Model A’s reach, from Q2.3, is 8. Sixteen is strictly larger than 8.

The measurement is centre-to-centre on the original 64×64 frame. `pixel_unshuffle(4)` folds each 4×4 patch into channels and leaves a 16×16 grid. Neighbouring cells on that grid have centres 4 original pixels apart. The fold does not bring in any cell beyond the patch that is already sitting there, and the 4×4 patch width is not added. `pixel_shuffle` is the same regrouping in reverse, and the 1×1 head mixes channels at one cell. Neither moves information to another cell.

Both ConvLSTM layers use a 3×3 kernel, stride 1, dilation 2, and padding 2. A 3×3 kernel with dilation 1 reads offsets −1, 0, and +1, so it moves one folded cell. Dilation 2 spaces those same three taps by 2, so the offsets are −2, 0, and +2. One application therefore moves information by 2 folded cells. Two folded cells are 2 × 4 = 8 original pixels. Padding 2 keeps the map at 16×16. At the border the kernel is cut off by the padding; the longest path is the one in the interior. The gate nonlinearities and the cell update are pointwise, so they do not add distance.

Inside one step τ there are three paths from the sources named in Q2.3 to the prediction \(\hat{x}_{\tau+1}\).

The new folded input \(u_\tau\) is concatenated with \(h^{(1)}_{\tau-1}\) and read by layer 1’s dilated 3×3. That produces \(h^{(1)}_\tau\). Layer 2 reads \(h^{(1)}_\tau\), not the previous hidden state, concatenated with \(h^{(2)}_{\tau-1}\), through its own dilated 3×3. The 1×1 convolution and `pixel_shuffle` then paint the frame. Two dilated 3×3 kernels, and nothing else that moves spatially: 2 cells + 2 cells = 4 folded cells, and 4 × 4 = 16 original pixels.

\(h^{(1)}_{\tau-1}\) enters that same layer-1 kernel and then the layer-2 kernel. It also travels 4 folded cells, so it also reaches 16 original pixels.

\(h^{(2)}_{\tau-1}\) enters only the layer-2 kernel, then the 1×1 head. That is 2 folded cells, 8 original pixels.

The cell state does not open a longer path. \(c_{\tau-1}\) is multiplied, at the same cell, by the forget gate. Its value does not shift onto a neighbour. The recurrence still matters: the previous hidden states are inputs to the convolutions, so memory can move, but only as far as the convolutions applied in this step. Because layer 2 consumes the new \(h^{(1)}_\tau\), the new frame and layer 1’s previous state are stacked through both kernels. Layer 2’s previous state sees only one kernel.

The longest path is therefore 16 original pixels. One path that attains it is \(u_\tau \rightarrow\) layer-1 3×3 dilation 2 \(\rightarrow h^{(1)}_\tau \rightarrow\) layer-2 3×3 dilation 2 \(\rightarrow \hat{x}_{\tau+1}\).

Model A’s longest path is those same two kernels with dilation 1. Each moves 1 folded cell: 1 + 1 = 2 cells, and 2 × 4 = 8 original pixels. Dilation 2 changes only the spacing of the taps. Each kernel’s step goes from 1 cell to 2 cells, the two kernels still stack, and 4 × 4 = 16.

## Parameters

Dilation does not add weights. Each layer is still one 3×3 convolution with bias, four gates, and 64 hidden channels.

Layer 1 reads 16 folded channels plus 64 hidden channels and writes 4 × 64 gate channels:

\((16 + 64) \times (4 \times 64) \times 3 \times 3 + (4 \times 64) = 80 \times 256 \times 9 + 256 = 184{,}576\).

Layer 2 reads 64 + 64 channels:

\((64 + 64) \times 256 \times 9 + 256 = 128 \times 256 \times 9 + 256 = 295{,}168\).

The 1×1 head maps 64 channels to 16:

\(64 \times 16 + 16 = 1{,}040\).

Total: \(184{,}576 + 295{,}168 + 1{,}040 = 480{,}784\).

The cap is \(1.5 \times 480{,}784 = 721{,}176\). 480,784 is under that cap. The counted total is `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Model A, built from the same cell with the default dilation 1, still counts 480,784.

## What was compared

Only this dilated ConvLSTM was implemented. A second candidate was not trained. No validation loss was measured. Videos 8000–8999 were not read for a comparison, and videos 9000–9999 were not read.
