import os
from os.path import join, basename

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import (get_scene_info,
                                                         save_image_crop)
from rastervision.pytorch_backend.examples.semantic_segmentation.utils import (
    example_multiband_transform, example_rgb_transform, imagenet_stats,
    Unnormalize)

# INRIA dataset in total contains 360 images of size 5000 x 5000 across 10 cities
TRAIN_CITIES = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
VAL_CITIES = ["vienna", "kitsap"]
# Debug
TRAIN_CITIES = ["austin"]
VAL_CITIES = ["vienna"]

TEST_CITIES = ["bellingham", "bloomington", "innsbruck", "sfo", "tyrol-e"]
NUM_IMAGES_PER_CITY = 1 #36

CLASS_NAMES = ["background", "building"]
CLASS_COLORS = ["white", "red"]

raw_uri = "/opt/data/inria/AerialImageDataset"
root_uri = "/opt/src/code/output"
nochip = False
augment = False
test= False

def get_config(runner,
               raw_uri: str,
               root_uri: str,
               augment: bool = False,
               nochip: bool = False,
               test: bool = False):

    channel_order = [0, 1, 2]
    channel_display_groups = None
    aug_transform = example_rgb_transform

    if augment:
        mu, std = imagenet_stats['mean'], imagenet_stats['std']
        mu, std = mu[channel_order], std[channel_order]

        base_transform = A.Normalize(mean=mu.tolist(), std=std.tolist())
        plot_transform = Unnormalize(mean=mu, std=std)

        aug_transform = A.to_dict(aug_transform)
        base_transform = A.to_dict(base_transform)
        plot_transform = A.to_dict(plot_transform)
    else:
        aug_transform = None
        base_transform = None
        plot_transform = None


    chip_sz = 256
    img_sz = chip_sz
    
    if nochip:
        chip_options = SemanticSegmentationChipOptions()
    else:
        chip_options = SemanticSegmentationChipOptions(
            window_method=SemanticSegmentationWindowMethod.sliding,
            stride=chip_sz)

    class_config = ClassConfig(names=CLASS_NAMES, colors=CLASS_COLORS)
    class_config.ensure_null_class()

    
    def make_scene(id) -> SceneConfig:
        raster_uri = f'{raw_uri}/train/images/{id}.tif'
        label_uri = f'{raw_uri}/train/gt/{id}.tif'


        raster_source = RasterioSourceConfig(
            uris=[raster_uri], channel_order=[0,1,2])

        label_source = SemanticSegmentationLabelSourceConfig(
            raster_source=RasterioSourceConfig(uris=[label_uri],
            transformers=[
                #ReclassTransformerConfig(mapping={255: 1})
                CastTransformerConfig(to_dtype='bool'),  # 255 --> True
                CastTransformerConfig(to_dtype='uint8')  # True --> 1
            ]))

        # URI will be injected by scene config.
        # Using rgb=True because we want prediction TIFFs to be in
        # RGB format.
        label_store = SemanticSegmentationLabelStoreConfig(
            rgb=True, vector_output=[PolygonVectorOutputConfig(class_id=0)])

        scene = SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            label_store=label_store)

        return scene

    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[make_scene(city + str(n)) for city in TRAIN_CITIES for n in range(1, NUM_IMAGES_PER_CITY + 1)],
        validation_scenes=[make_scene(city + str(n)) for city in VAL_CITIES for n in range(1, NUM_IMAGES_PER_CITY + 1)])


    if nochip:
        window_opts = {}
        # set window configs for training scenes
        for s in scene_dataset.train_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                # method=GeoDataWindowMethod.sliding,
                method=GeoDataWindowMethod.random,
                size=img_sz,
                # size_lims=(200, 300),
                h_lims=(200, 300),
                w_lims=(200, 300),
                max_windows=2209,
            )
        # set window configs for validation scenes
        for s in scene_dataset.validation_scenes:
            window_opts[s.id] = GeoDataWindowConfig(
                method=GeoDataWindowMethod.sliding,
                size=img_sz,
                stride=img_sz // 2)

        data = SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=window_opts,
            img_sz=img_sz,
            img_channels=len(channel_order),
            num_workers=4,
            channel_display_groups=channel_display_groups)
    else:
        data = SemanticSegmentationImageDataConfig(
            img_sz=img_sz,
            img_channels=len(channel_order),
            num_workers=4,
            channel_display_groups=channel_display_groups,
            base_transform=base_transform,
            aug_transform=aug_transform,
            plot_options=PlotOptions(transform=plot_transform))

    model = SemanticSegmentationModelConfig(backbone=Backbone.resnet50)

    backend = PyTorchSemanticSegmentationConfig(
        data=data,
        model=model,
        solver=SolverConfig(
            lr=1e-4,
            num_epochs=5,
            test_num_epochs=2,
            batch_sz=8,
            test_batch_sz=2,
            one_cycle=True),
        log_tensorboard=True,
        run_tensorboard=False,
        test_mode=test)

    pipeline = SemanticSegmentationConfig(
        root_uri=root_uri,
        dataset=scene_dataset,
        backend=backend,
        channel_display_groups=channel_display_groups,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz,
        chip_options=chip_options)

    return pipeline



# if __name__ == "__main__":
#     import pdb; pdb.set_trace()
