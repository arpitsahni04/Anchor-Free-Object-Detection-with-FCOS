# Credit to Justin Johnsons' EECS-598 course at the University of Michigan,
# from which this assignment is heavily drawn.
import math
from typing import Dict, List, Optional

import torch
from detection_utils import *
from torch import nn
from torch.nn import functional as F
from torch.utils.data._utils.collate import default_collate
from torchvision.ops import sigmoid_focal_loss
from torchvision import models
from torchvision.models import feature_extraction


class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    NOTE: We could use any convolutional network architecture that progressively
    downsamples the input image and couple it with FPN. We use a small enough
    backbone that can work with Colab GPU and get decent enough performance.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        # Torchvision models only return features from the last level. Detector
        # backbones (with FPN) require intermediate features of different scales.
        # So we wrap the ConvNet with torchvision's feature extractor. Here we
        # will get output features with names (c3, c4, c5) with same stride as
        # (p3, p4, p5) described above.
        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        # Pass a dummy batch of input images to infer shapes of (c3, c4, c5).
        # Features are a dictionary with keys as defined above. Values are
        # batches of tensors in NCHW format, that give intermediate features
        # from the backbone network.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        in_channels_list = [o.shape[1] for o in dummy_out.values()]
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]
        self.dummy_shapes = dummy_out_shapes

        # print("For dummy input images with shape: (2, 3, 224, 224)")
        # for level_name, feature_shape in dummy_out_shapes:
        #     print(f"Shape of {level_name} features: {feature_shape}")
        

        c3_out =dummy_out_shapes[0][1][1]
        c4_out = dummy_out_shapes[1][1][1]
        c5_out =dummy_out_shapes[2][1][1]
        self.fpn_params = nn.ModuleDict()
        self.fpn_params["p3_cv1"] = nn.Conv2d(c3_out, self.out_channels, kernel_size=1)
        self.fpn_params["p4_cv1"] = nn.Conv2d(c4_out, self.out_channels, kernel_size=1)
        self.fpn_params["p5_cv1"] = nn.Conv2d(c5_out, self.out_channels,kernel_size =1)
        self.fpn_params["p3_cv3"] = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1) # to compensate for upsample
        self.fpn_params["p4_cv3"] = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.fpn_params["p5_cv3"] = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        
        


    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}

        feats_p3_lat = self.fpn_params["p3_cv1"](backbone_feats["c3"])
        feats_p4_lat = self.fpn_params["p4_cv1"](backbone_feats["c4"])
        feats_p5_lat = self.fpn_params["p5_cv1"](backbone_feats["c5"])
        fpn_feats["p5"] = self.fpn_params["p5_cv3"](feats_p5_lat)
        
        feats_p4_lat = feats_p4_lat + F.interpolate(feats_p5_lat, scale_factor=2)
        fpn_feats["p4"] = self.fpn_params["p4_cv3"](feats_p4_lat)
        
        feats_p3_lat = feats_p3_lat + F.interpolate(feats_p4_lat, scale_factor=2)
        fpn_feats["p3"] = self.fpn_params["p3_cv3"](feats_p3_lat)


        return fpn_feats

class FCOSPredictionNetwork(nn.Module):
    """
    FCOS prediction network that accepts FPN feature maps from different levels
    and makes three predictions at every location: bounding boxes, class ID and
    centerness. This module contains a "stem" of convolution layers, along with
    one final layer per prediction. For a visual depiction, see Figure 2 (right
    side) in FCOS paper: https://arxiv.org/abs/1904.01355

    We will use feature maps from FPN levels (P3, P4, P5) and exclude (P6, P7).
    """

    def __init__(
        self, num_classes: int, in_channels: int, stem_channels: List[int]
    ):
        """
        Args:
            num_classes: Number of object classes for classification.
            in_channels: Number of channels in input feature maps. This value
                is same as the output channels of FPN, since the head directly
                operates on them.
            stem_channels: List of integers giving the number of output channels
                in each convolution layer of stem layers.
        """
        super().__init__()

        # Fill these.
        stem_cls = []
        stem_box = []
        # Replace "pass" statement with your code
        for out_channels in stem_channels:
            stem_cls.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
            stem_cls.append(nn.ReLU())
            stem_box.append(nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
            stem_box.append(nn.ReLU())
            in_channels = out_channels       

        # Wrap the layers defined by student into a `nn.Sequential` module:
        self.stem_cls = nn.Sequential(*stem_cls)
        self.stem_box = nn.Sequential(*stem_box)

        # Initialize all layers.
        for stems in (self.stem_cls, self.stem_box):
            for layer in stems:
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)



        # Replace these lines with your code, keep variable names unchanged.
        self.pred_cls =nn.Conv2d(stem_channels[-1], num_classes, 3, stride=1, padding=1)    # Class prediction conv
        self.pred_box = nn.Conv2d(stem_channels[-1], 4, 3, stride=1, padding=1)    # Box regression conv
        self.pred_ctr =nn.Conv2d(stem_channels[-1], 1, 3, stride=1, padding=1)   # Centerness conv

        if self.pred_cls is not None:
            torch.nn.init.constant_(self.pred_cls.bias, -math.log(99))

    def forward(self, feats_per_fpn_level: TensorDict) -> List[TensorDict]:
        """
        Accept FPN feature maps and predict the desired outputs at every location
        (as described above). Format them such that channels are placed at the
        last dimension, and (H, W) are flattened (having channels at last is
        convenient for computing loss as well as perforning inference).

        Args:
            feats_per_fpn_level: Features from FPN, keys {"p3", "p4", "p5"}. Each
                tensor will have shape `(batch_size, fpn_channels, H, W)`. For an
                input (224, 224) image, H = W are (28, 14, 7) for (p3, p4, p5).

        Returns:
            List of dictionaries, each having keys {"p3", "p4", "p5"}:
            1. Classification logits: `(batch_size, H * W, num_classes)`.
            2. Box regression deltas: `(batch_size, H * W, 4)`
            3. Centerness logits:     `(batch_size, H * W, 1)`
        """


        class_logits = {}
        boxreg_deltas = {}
        centerness_logits = {}
        for level_name  in feats_per_fpn_level.keys():
            class_logits[level_name] = self.pred_cls(self.stem_cls(feats_per_fpn_level[level_name]))
            class_logits[level_name] = class_logits[level_name].flatten(2).permute(0, 2, 1)
            boxreg_deltas[level_name] = self.pred_box(self.stem_box(feats_per_fpn_level[level_name]))
            boxreg_deltas[level_name] = boxreg_deltas[level_name].flatten(2).permute(0, 2, 1)
            centerness_logits[level_name] = self.pred_ctr(self.stem_box(feats_per_fpn_level[level_name]))
            centerness_logits[level_name] = centerness_logits[level_name].flatten(2).permute(0, 2, 1)


        return [class_logits, boxreg_deltas, centerness_logits]

class FCOS(nn.Module):
    """
    FCOS: Fully-Convolutional One-Stage Detector

    This class puts together everything you implemented so far. It contains a
    backbone with FPN, and prediction layers (head). It computes loss during
    training and predicts boxes during inference.
    """

    def __init__(
        self, num_classes: int, fpn_channels: int, stem_channels: List[int]
    ):
        super().__init__()
        self.num_classes = num_classes


        self.backbone = DetectorBackboneWithFPN(out_channels=fpn_channels)
        self.pred_net = FCOSPredictionNetwork(num_classes,fpn_channels,stem_channels)

        self._normalizer = 150  # per image

    def forward(
        self,
        images: torch.Tensor,
        gt_boxes: Optional[torch.Tensor] = None,
        test_score_thresh: Optional[float] = None,
        test_nms_thresh: Optional[float] = None,
    ):
        """
        Args:
            images: Batch of images, tensors of shape `(B, C, H, W)`.
            gt_boxes: Batch of training boxes, tensors of shape `(B, N, 5)`.
                `gt_boxes[i, j] = (x1, y1, x2, y2, C)` gives information about
                the `j`th object in `images[i]`. The position of the top-left
                corner of the box is `(x1, y1)` and the position of bottom-right
                corner of the box is `(x2, x2)`. These coordinates are
                real-valued in `[H, W]`. `C` is an integer giving the category
                label for this bounding box. Not provided during inference.
            test_score_thresh: During inference, discard predictions with a
                confidence score less than this value. Ignored during training.
            test_nms_thresh: IoU threshold for NMS during inference. Ignored
                during training.

        Returns:
            Losses during training and predictions during inference.
        """


        backbone_feats = self.backbone(images)
        pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = self.pred_net(backbone_feats)


        shape_per_fpn_level ={}
        for name, feat in backbone_feats.items():
            shape_per_fpn_level[name]=feat.shape 
        
        locations_per_fpn_level = get_fpn_location_coords(shape_per_fpn_level,self.backbone.fpn_strides,device = images.device)


        if not self.training:
            # During inference, just go to this method and skip rest of the
            # forward pass.
            # fmt: off
            return self.inference(
                images, locations_per_fpn_level,
                pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits,
                test_score_thresh=test_score_thresh,
                test_nms_thresh=test_nms_thresh,
            )
            # fmt: on

        
        matched_gt_boxes = []
        matched_gt_deltas = []
        for idx in range(gt_boxes.shape[0]):
            matched_loc = fcos_match_locations_to_gt(locations_per_fpn_level,self.backbone.fpn_strides,gt_boxes[idx])
            matched_gt_boxes.append(matched_loc)
        # Calculate GT deltas for these matched boxes. Similar structure
        # as `matched_gt_boxes` above. Fill this list:
        for idx in range(gt_boxes.shape[0]):
            matched_gt_deltas_dict = {}
            for level, feat_loc in locations_per_fpn_level.items():
                 matched_gt_deltas_dict[level] = fcos_get_deltas_from_locations(feat_loc, 
                                                                                   matched_gt_boxes[idx][level] , 
                                                                                    self.backbone.fpn_strides[level])
            matched_gt_deltas.append(matched_gt_deltas_dict)  

        matched_gt_boxes = default_collate(matched_gt_boxes)
        matched_gt_deltas = default_collate(matched_gt_deltas)

        # Combine predictions and GT from across all FPN levels.
        # shape: (batch_size, num_locations_across_fpn_levels, ...)
        matched_gt_boxes = self._cat_across_fpn_levels(matched_gt_boxes)
        matched_gt_deltas = self._cat_across_fpn_levels(matched_gt_deltas)
        pred_cls_logits = self._cat_across_fpn_levels(pred_cls_logits)
        pred_boxreg_deltas = self._cat_across_fpn_levels(pred_boxreg_deltas)
        pred_ctr_logits = self._cat_across_fpn_levels(pred_ctr_logits)

        # Perform EMA update of normalizer by number of positive locations.
        num_pos_locations = (matched_gt_boxes[:, :, 4] != -1).sum()
        pos_loc_per_image = num_pos_locations.item() / images.shape[0]
        self._normalizer = 0.9 * self._normalizer + 0.1 * pos_loc_per_image


        pred_ctr_logits = pred_ctr_logits.flatten()


        
        matched_gt_deltas = matched_gt_deltas.cuda()

        loss_box =  (0.25) * F.l1_loss(pred_boxreg_deltas,matched_gt_deltas,reduction="none")
        # make background 0
        mask_neg = matched_gt_deltas<0
        loss_box[mask_neg] *=0.0
        
        matched_ctr =fcos_make_centerness_targets(matched_gt_deltas.view(-1, 4)) ## doubt 2 Centerness loss flattening 
        loss_ctr =  F.binary_cross_entropy_with_logits(pred_ctr_logits,matched_ctr,reduction="none")
        loss_ctr[matched_ctr<=0] *= 0.0  
        
        idxs = matched_gt_boxes[..., -1].to(torch.int64)
        
        one_hotted = torch.cat((torch.zeros(1, 20),torch.eye(20)), dim = 0).cuda()
        idxs+=1 
        gt_cls_logits = one_hotted[idxs]
        loss_cls = sigmoid_focal_loss(pred_cls_logits, gt_cls_logits)
        
    

        return {
            "loss_cls": loss_cls.sum() / (self._normalizer * images.shape[0]),
            "loss_box": loss_box.sum() / (self._normalizer * images.shape[0]),
            "loss_ctr": loss_ctr.sum() / (self._normalizer * images.shape[0]),
        }

    @staticmethod
    def _cat_across_fpn_levels(
        dict_with_fpn_levels: Dict[str, torch.Tensor], dim: int = 1
    ):
        """
        Convert a dict of tensors across FPN levels {"p3", "p4", "p5"} to a
        single tensor. Values could be anything - batches of image features,
        GT targets, etc.
        """
        return torch.cat(list(dict_with_fpn_levels.values()), dim=dim)

    def inference(
        self,
        images: torch.Tensor,
        locations_per_fpn_level: Dict[str, torch.Tensor],
        pred_cls_logits: Dict[str, torch.Tensor],
        pred_boxreg_deltas: Dict[str, torch.Tensor],
        pred_ctr_logits: Dict[str, torch.Tensor],
        test_score_thresh: float = 0.3,
        test_nms_thresh: float = 0.5,
    ):
        """
        Run inference on a single input image (batch size = 1). Other input
        arguments are same as those computed in `forward` method. This method
        should not be called from anywhere except from inside `forward`.

        Returns:
            Three tensors:
                - pred_boxes: Tensor of shape `(N, 4)` giving *absolute* XYXY
                  co-ordinates of predicted boxes.

                - pred_classes: Tensor of shape `(N, )` giving predicted class
                  labels for these boxes (one of `num_classes` labels). Make
                  sure there are no background predictions (-1).

                - pred_scores: Tensor of shape `(N, )` giving confidence scores
                  for predictions: these values are `sqrt(class_prob * ctrness)`
                  where class_prob and ctrness are obtained by applying sigmoid
                  to corresponding logits.
        """

        # Gather scores and boxes from all FPN levels in this list. Once
        # gathered, we will perform NMS to filter highly overlapping predictions.
        pred_boxes_all_levels = []
        pred_classes_all_levels = []
        pred_scores_all_levels = []

        for level_name in locations_per_fpn_level.keys():

            # Get locations and predictions from a single level.
            # We index predictions by `[0]` to remove batch dimension.
            level_locations = locations_per_fpn_level[level_name]
            level_cls_logits = pred_cls_logits[level_name][0]
            level_deltas = pred_boxreg_deltas[level_name][0]
            level_ctr_logits = pred_ctr_logits[level_name][0]


            level_pred_boxes, level_pred_classes, level_pred_scores = (
                None,
                None,
                None,  # Need tensors of shape: (N, 4) (N, ) (N, )
            )

            # Compute geometric mean of class logits and centerness:
            level_pred_scores = torch.sqrt(
                level_cls_logits.sigmoid_() * level_ctr_logits.sigmoid_()
            )
            
            # Step 1:
            # Replace "pass" statement with your code
            level_pred_scores,level_pred_classes = torch.max(level_pred_scores, dim=1)
            
            # Step 2:
            idx = level_pred_scores > test_score_thresh
            idx = idx.nonzero()
            level_pred_classes = level_pred_classes[idx].flatten().cuda()
            level_pred_scores = level_pred_scores[idx].flatten().cuda()

            
            level_deltas = level_deltas[idx].view(-1,4)
            level_locations = level_locations[idx].view(-1,2)
            level_pred_boxes = fcos_apply_deltas_to_locations(level_deltas,
                                                              level_locations,
                                                              self.backbone.fpn_strides[level_name])
            
            # Step 4: Use `images` to get (height, width) for clipping.
            # Replace "pass" statement with your code
            H, W = images.shape[2], images.shape[3]
            level_pred_boxes[:,0] = level_pred_boxes[:,0].clip(min=0)
            level_pred_boxes[:,1] = level_pred_boxes[:,1].clip(min=0)
            level_pred_boxes[:,2] = level_pred_boxes[:,2].clip(max=H)
            level_pred_boxes[:,3] = level_pred_boxes[:,3].clip(max=W)



            pred_boxes_all_levels.append(level_pred_boxes.cuda())
            pred_classes_all_levels.append(level_pred_classes.cuda())
            pred_scores_all_levels.append(level_pred_scores.cuda())

        pred_boxes_all_levels = torch.cat(pred_boxes_all_levels)
        pred_classes_all_levels = torch.cat(pred_classes_all_levels)
        pred_scores_all_levels = torch.cat(pred_scores_all_levels)

        keep = class_spec_nms(
            pred_boxes_all_levels,
            pred_scores_all_levels,
            pred_classes_all_levels,
            iou_threshold=test_nms_thresh,
        )
        pred_boxes_all_levels = pred_boxes_all_levels[keep]
        pred_classes_all_levels = pred_classes_all_levels[keep]
        pred_scores_all_levels = pred_scores_all_levels[keep]
        return (
            pred_boxes_all_levels,
            pred_classes_all_levels,
            pred_scores_all_levels,
        )
