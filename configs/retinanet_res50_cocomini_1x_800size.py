# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from detection import models


class CustomRetinaNetConfig(models.RetinaNetConfig):
    def __init__(self):
        super().__init__()

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="cocomini",
            root="train",
            ann_file="annotations/cocomini.json",
            remove_images_without_annotations=True,
        )

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = 600
        self.nr_images_epoch = 10000
        self.test_dataset = dict(
            name="cocomini",
            root="val2017",
            ann_file="annotations/instances_val2017.json",
        )


Net = models.RetinaNet
Cfg = CustomRetinaNetConfig
