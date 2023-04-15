def ceiln(num, n):
    import math

    assert isinstance(n, int)
    if num % n == 0:
        return num
    return round(math.ceil(num / n) * n)

def save_imgs(imgs, fname, root=''):
    import os
    import torch
    import pathlib
    from torchvision.utils import save_image

    os.makedirs(root, exist_ok=True)
    nrows = imgs[0].size(0)
    imgs = torch.cat(imgs, dim=0)
    save_image(imgs, pathlib.Path(root) / fname, nrow=nrows)


def variance_scaling_init_(tensor, scale=1.):
    import math
    from torch import nn

    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)  # noqa
    n = (fan_in + fan_out) / 2.
    limit = math.sqrt(3. * scale / n)

    # smaller than this would cause mostly zero init due to amp
    assert limit > 1e-7, limit
    return nn.init.uniform_(tensor, -limit, limit)


def seed_all(seed):
    import os
    import torch
    import random
    import numpy as np

    if torch.cuda.is_available():
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        from torch.backends import cudnn
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.set_deterministic_debug_mode('warn')
    torch.use_deterministic_algorithms(True, warn_only=True)
