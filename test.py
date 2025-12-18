data_path = 'kaggle/input/cub200crops-224x224/cub200_crops_224x224/'
train_dir = data_path + 'train_cropped_augmented/'
test_dir = data_path + 'test_cropped/'
train_push_dir = data_path + 'train_cropped/'
num_classes = 20
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

img_size = 196
learning_rate = 1e-4
train_batch_size = 16
test_batch_size = 16
num_warm_epochs = 1
num_train_epochs = 20

train_push_batch_size = 16

join_optimizer_lrs = {'features': 1e-4, 'add_on_layers': 1e-3, 'prototype_vectors': 1e-3}
join_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 1e-6, 'prototype_vectors': 1e-6}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1.0,
    'clst': 0.0001,
    'sep': 0.0001,
    'l1': 0
}

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 ==0]
num_last_layer_iterations = 5

prototype_shape = (num_classes * 10, 128, 1, 1)
base_architecture = 'resnet18'

model_dir = './saved_model/'
analasis_output_dir = './analysis_output'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(analasis_output_dir):
    os.makedirs(analasis_output_dir)

# Normalization parameters frmo process.py
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
seed = 42
#data loading
def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)


# define transforms for training and validation sets
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
val_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# create dataset from the preprocessed directories
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4,
    worker_init_fn=worker_init_fn, pin_memory=False)
val_loader = DataLoader(
    val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4,
    worker_init_fn=worker_init_fn, pin_memory=False)

print(f'Number of training samples: {len(train_dataset)}')
print(f'Number of validation samples: {len(val_dataset)}')

def undo_preprocess(x):
    """
    Undo the normalization so the image can be visualized."""
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x

def save_image(img_tensor, fname):
    """
    Undo normalization and save a tensor as an image file.
    """
    img = undo_preprocess(img_tensor.clone().squeeze(0).cpu().numpy())
    img = np.transpose(img, (1, 2, 0))  # C x H x W to H x W x C
    plt.imsave(fname, img)

def makedir(path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def compute_proto_layer_rf_info(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):
    """
    This is the original version (if used in parts of the code)
    """
    rf = 1
    j = 1
    start = 0.5
    for f, s, p in zip(layer_filter_sizes, layer_strides, layer_paddings):
        rf = rf + (f - 1) * j
        start = start + ((f - 1) / 2 - p) * j
        j = j * s
    return {'rf': rf, 'j': j, 'start': start, 'prototype_kernel_size': prototype_kernel_size}

def copute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):
    """
    This is a refined version used in our PPNet constructor"""
    rf = 1
    j = 1
    start = 0.5
    for f, s, p in zip(layer_filter_sizes, layer_strides, layer_paddings):
        rf = rf + (f - 1) * j
        start = start + ((f - 1) / 2 - p) * j
        j = j * s
    return {'rf': rf, 'j': j, 'start': start, 'prototype_kernel_size': prototype_kernel_size}

def compute_rf_prototyoes(img_size, proto_rf_info, spatial_location):
    """
    Compute the center position in the input image correesponding to the given spatial location in the prototype layer.
    """
    center_h = proto_rf_info['start'] + spatial_location[0] * proto_rf_info['j']
    center_w = proto_rf_info['start'] + spatial_location[1] * proto_rf_info['j']
    # The effective receptive field computed from the base network
    rf = proto_rf_info['rf']
    # incorporate the prototype kernel size into the receptive field
    prototype_kernel_size = proto_rf_info.get('prototype_kernel_size', 1)
    additional = (prototype_kernel_size - 1) / 2.0
    half_rf = (rf / 2.0) + additional
    # computen boundaries
    h_start = int(round(center_h - half_rf))
    h_end = int(round(center_h + half_rf))
    w_start = int(round(center_w - half_rf))
    w_end = int(round(center_w + half_rf))
    # Clamp to the image boundaries
    h_start = max(h_start, 0)
    w_start = max(w_start, 0)
    h_end = min(h_end, img_size)
    w_end = min(w_end, img_size)
    return h_start, h_end, w_start, w_end

def compute_rf_proto_at_spatial_location(proto_rf_info, spatial_location):
    """
    Alternative function: given a spatial location in the prototype layer, return its receptive field.
    """
    return compute_rf_prototyoes(None, proto_rf_info, spatial_location)

# VGG_feature Extractor
cfg_E = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

class VGG_features(nn.Module):
    """
    Custom implementation of VGG-based feature extractor.
    
    This builds the convolutional base from a VGG-like configuration.
    Optionally includes batch normalization and returns the kernel info for receptive field tracking.
    """
    def __init__(self, cfg, batch_norm=False, init_weights=True):
        super(VGG_features, self).__init__()
        self.batch_norm = batch_norm
        self.kernel_sizes = []
        self.strides = []
        self.paddings = []
        self.features = self._make_layers(cfg, batch_norm)
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        return self.features(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.kernel_sizes.append(2)
                self.strides.append(2)
                self.paddings.append(0)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers.extend([conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv2d, nn.ReLU(inplace=True)])
                self.kernel_sizes.append(3)
                self.strides.append(1)
                self.paddings.append(1)
                in_channels = v
        return nn.Sequential(*layers)
    
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings
    
def vgg11_features(pretrained=False, **kwargs):
    # implementation of VGG11: skipping prretrained eight loading for simplicity
    model = VGG_features(cfg_E, batch_norm=False, **kwargs)
    return model
