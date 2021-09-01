from typing import List, Tuple
from h5dataloader.pytorch import HDF5Dataset
from h5dataloader.common.augmentation import *
from h5dataloader.common.crop import *
from h5dataloader.common.resize import *
from pointsmap import Points, VoxelGridMap

IMG_HEIGHT = 256
IMG_WIDTH = 512

class TrainHDF5Dataset(HDF5Dataset):
    def __init__(
        self,
        h5_paths: List[str] = [],
        config: str = None,
        quiet: bool = False,
        block_size: int = 0,
        use_mods: Tuple[int, int] = None,
        visibility_filter_radius: int = 0,
        visibility_filter_threshold: float = 3.0,
        jitter: float = 0.1
    ) -> None:
        super().__init__(h5_paths, config, quiet, block_size, use_mods, visibility_filter_radius, visibility_filter_threshold)

        self.adjust_brightness = Adjust_brightness(factor_range=ValueRange(max(0, 1 - jitter), 1 + jitter))
        self.adjust_contrast = Adjust_contrast(factor_range=ValueRange(max(0, 1 - jitter), 1 + jitter))
        self.adjust_saturation = Adjust_saturation(factor_range=ValueRange(max(0, 1 - jitter), 1 + jitter))
        self.resize_bilinear = Resize(output_size=(IMG_HEIGHT, IMG_WIDTH), interpolation=INTER_LINEAR)
        self.resize_nn = Resize(output_size=(IMG_HEIGHT, IMG_WIDTH), interpolation=INTER_NEAREST)
        self.flip_horizontal = Flip_2d(hflip_rate=0.5, vflip_rate=0.0)
        self.random_pose = RandomPose(range_tr=(0.6, 1.3, 0.7), range_rot=3.0)

    def create_rgb_rgb(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_BGR8, key, link_idx, minibatch_config)
        src: Data = Data(data=self.h5links[h5_key][()], type=self.h5links[h5_key].attrs[H5_ATTR_TYPE])
        adjusted_b = self.adjust_brightness(h5_key, src)
        adjusted_c = self.adjust_contrast(h5_key, adjusted_b)
        adjusted_s = self.adjust_saturation(h5_key, adjusted_c)
        resized = self.resize_bilinear(h5_key, adjusted_s)
        fliped = self.flip_horizontal(h5_key, resized)
        return fliped.data

    def create_d_sparse(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_VOXEL_SEMANTIC3D, key, link_idx, minibatch_config)
        vgm = VoxelGridMap(quiet=self.quiet)
        vgm.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        vgm.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        vgm.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        vgm.set_empty_voxelgridmap(
            self.h5links[h5_key].shape, self.h5links[h5_key].attrs[H5_ATTR_VOXELSIZE],
            tuple(self.h5links[h5_key].attrs[H5_ATTR_VOXELMIN]), tuple(self.h5links[h5_key].attrs[H5_ATTR_VOXELMAX]),
            tuple(self.h5links[h5_key].attrs[H5_ATTR_VOXELCENTER]), tuple(self.h5links[h5_key].attrs[H5_ATTR_VOXELORIGIN].tolist())
        )

        pose = Data(data=self.create_pose_from_pose(key, link_idx, minibatch_config), type=TYPE_POSE)
        pose = self.random_pose(h5_key, pose)
        idxs = vgm.get_voxels_include_frustum(translation=pose.data[:3], quaternion=pose.data[-4:])
        vgm.set_voxels(idxs, self.h5links[h5_key][()][idxs])

        src = Data(
            data=vgm.create_depthmap(
                translation=pose.data[:3], quaternion=pose.data[-4:],
                filter_radius=self.visibility_filter_radius,
                filter_threshold=self.visibility_filter_threshold
            ),
            type=TYPE_DEPTH
        )
        fliped = self.flip_horizontal(h5_key, src)
        return fliped.data

    def create_gt_target(self, key: str, link_idx: int, minibatch_config: dict) -> np.ndarray:
        h5_key: str = self.create_h5_key(TYPE_POINTS)
        pts = Points(quiet=self.quiet)
        pts.set_intrinsic(self.create_intrinsic_array(key, link_idx, minibatch_config))
        pts.set_shape(minibatch_config[CONFIG_TAG_SHAPE])
        pts.set_depth_range(minibatch_config[CONFIG_TAG_RANGE])
        pts.set_points(self.h5links[h5_key][()])

        pose = Data(data=self.create_pose_from_pose(key, link_idx, minibatch_config), type=TYPE_POSE)
        # pose = self.random_pose(h5_key, pose)

        src = Data(
            data=pts.create_depthmap(
                translation=pose.data[:3], quaternion=pose.data[-4:],
                filter_radius=self.visibility_filter_radius,
                filter_threshold=self.visibility_filter_threshold
            ),
            type=TYPE_DEPTH
        )
        fliped = self.flip_horizontal(h5_key, src)
        return fliped.data
