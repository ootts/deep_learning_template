from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.input.min_size_train
        max_size = cfg.input.max_size_train
        horizontal_flip_prob = cfg.input.horizontal_flip_prob
        brightness = cfg.input.brightness
        contrast = cfg.input.contrast
        saturation = cfg.input.saturation
        hue = cfg.input.hue
    else:
        min_size = cfg.input.min_size_test
        max_size = cfg.input.max_size_test
        horizontal_flip_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    normalize_transform = T.Normalize(
        mean=cfg.input.pixel_mean, std=cfg.input.pixel_std
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    ts = [
        T.Resize(min_size, max_size),
        color_jitter,
        T.RandomHorizontalFlip(horizontal_flip_prob),
        T.ToTensor(),
    ]
    if cfg.input.do_normalize:
        ts.append(normalize_transform)
    transform = T.Compose(ts)
    return transform
