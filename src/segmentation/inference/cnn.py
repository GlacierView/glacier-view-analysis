"""U-Net with a ResNet-50 encoder — the glacier segmentation model.

This is the single definition of the model. `infer.py`, `final_areas.py` and
`training/src/train.py` all import `UNet` and `conv_block` from here; do not
copy the classes elsewhere.

Checkpoints are saved with `torch.save(model)`, i.e. as pickled modules rather
than state dicts, so the pickle records a reference to these class names. Two
consequences:

  * Changing a layer's name or shape can stop an existing checkpoint loading.
  * Checkpoints written before this module existed reference `__main__.UNet`
    and `__main__.conv_block`, so a script that loads one must import both
    names into its own globals (which is what the callers do) and run as
    `__main__`.

Input is (N, IN_CHANNELS, 128, 128), channels in this order:

    0 NDWI   1 NDSI   2 blue   3 green   4 red
    5 nir    6 swir   7 thermal   8 DEM

`swir_2` is deliberately excluded for quality reasons. The two spectral indices
are derived from the raw bands and prepended by the caller.
"""

import torch
import torch.nn as nn
import torchvision.models as models

#: Number of input channels the model expects. Both released checkpoints were
#: trained at 9; verified against their serialized tensors.
IN_CHANNELS = 9


class conv_block(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.bn1(self.conv1(inputs)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """ResNet-50-encoder U-Net producing a 2-class segmentation map.

    Args:
        n_class: number of output classes (2: background, glacier).
        freeze_encoder: if True, hold the pretrained ResNet weights fixed and
            train only the replaced stem and the decoder.
    """

    def __init__(self, n_class=2, freeze_encoder=True):
        super().__init__()
        self.n_class = n_class
        self.freeze_encoder = freeze_encoder

        # ImageNet-pretrained ResNet-50 with its classifier head (the final
        # avgpool and fc layers) removed, leaving eight sequential stages.
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        if freeze_encoder:
            # Skip index 0: the stem is replaced below and must stay trainable.
            for i, param in enumerate(resnet.parameters()):
                if i != 0:
                    param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # ResNet expects 3 channels; we feed IN_CHANNELS. Randomly initialised.
        self.resnet[0] = nn.Conv2d(
            IN_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

        # Decoder. Each stage upsamples, concatenates the matching encoder
        # feature map (the U-Net skip connection), then convolves back down.
        # The final stage concatenates the raw input, hence 32 + IN_CHANNELS.
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.c1 = conv_block(2048, 1024)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.c2 = conv_block(1024, 512)

        self.deconv3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.c3 = conv_block(512, 256)

        self.deconv4 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.c4 = conv_block(128, 64)

        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.c5 = conv_block(32 + IN_CHANNELS, 16)

        self.classifier = nn.Conv2d(16, self.n_class, kernel_size=1)

        # Unused by forward(). Retained so that the constructed module matches
        # the structure of the released checkpoints — removing them would make
        # a state_dict extracted from those checkpoints fail a strict load.
        self.deconv6 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(16)
        self.deconv7 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        self.deconv8 = nn.ConvTranspose2d(16, 4, 3, stride=2, padding=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(4)
        self.c8 = nn.Conv2d(4, 4, 6, stride=8, padding=0)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        # Run the encoder, keeping every intermediate feature map so the
        # decoder can reach back for its skip connections. feats[0] is the raw
        # input and feats[i + 1] is the output of self.resnet[i].
        feats = [images]
        for stage in self.resnet:
            feats.append(stage(feats[-1]))

        # Skips pair with the decoder stages below, coarsest first. The last
        # one is the raw input, reattached at full resolution.
        skips = [feats[7], feats[6], feats[5], feats[3], feats[0]]
        stages = [
            (self.deconv1, self.bn1, self.c1),
            (self.deconv2, self.bn2, self.c2),
            (self.deconv3, self.bn3, self.c3),
            (self.deconv4, self.bn4, self.c4),
            (self.deconv5, self.bn5, self.c5),
        ]

        y = feats[-1]
        for (deconv, bn, conv), skip in zip(stages, skips, strict=True):
            y = bn(self.relu(deconv(y)))
            y = conv(torch.cat([y, skip], dim=1))

        # Logits, not probabilities: callers apply softmax themselves.
        return self.classifier(y)
