import torch 
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from Classifier import Conv_network as pers_cls
import os



# Prefer a GUI backend when one is available, but fall back cleanly in
# headless environments where Qt is not installed.
for backend in ('MacOSX', 'QtAgg', 'TkAgg', 'GTK3Agg', 'Agg'):
    try:
        mpl.use(backend, force=True)
        break
    except Exception:
        continue

def prepare_for_model(tensor):
    """Convert a hand-drawn 256 x 256 image into a model-ready tensor.

    MNIST images are 28 x 28 grayscale images. The drawing canvas is larger,
    so the image is resized and then given the batch and channel dimensions
    expected by the neural network: (1, 1, 28, 28).
    """

    tensor = np.clip(tensor, 0,1 )

    downscaled = ndimage.zoom(tensor, 28/256, order=1)
    
    # Bilinear interpolation (order=1) reduces the canvas to MNIST size.
    downscaled = np.clip(downscaled, 0,1)

    # Add one dimension for the channel and one for the batch.
    ready_tsr = torch.from_numpy(downscaled).unsqueeze(0).unsqueeze(0)

    return ready_tsr


# This allows testing by drawing digits and sending the tensors (after processing) to the model to evaluate it
class Digitdrawer:
    """Provide a matplotlib canvas for drawing a digit by hand."""

    def __init__(self):
        # Store the drawing as a 256 x 256 grayscale image.
        self.canvas = np.zeros((256,256), dtype=np.float32)
        self.fig, self.ax = plt.subplots(figsize=(4,4))
        self.img_plot = self.ax.imshow(self.canvas,cmap="gray", vmin=0, vmax=1)
        self.ax.set_title('Draw a digit and close the tab to save')
        self.drawing = False

        self.fig.canvas.mpl_connect('button_press_event', self.press)
        self.fig.canvas.mpl_connect('button_release_event', self.release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.move)
        self.radius = 20
        self.sigma = self.radius/4
        self.shade_grad = self.radius

        
        coords_tensor = torch.arange(-self.radius, self.radius)
        coords_tensor = torch.meshgrid(coords_tensor, coords_tensor, indexing='xy')
        coords_tensor = torch.stack(coords_tensor, dim=0)
        mark_array = np.clip((coords_tensor[0]**2+coords_tensor[1]**2)**0.5, a_max=self.radius, a_min=0)
        mark_array = (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((mark_array - 0) / self.sigma) ** 2)*self.shade_grad
                
        mark_array = torch.maximum(mark_array, torch.tensor(0))
        self.soft_brush = mark_array.numpy()
        plt.show()

    def press(self, event):
        # Start drawing only when the mouse press occurs inside the canvas.
        if event.inaxes == self.ax:
            self.drawing = True
            self.draw(event)

    def move(self, event):
        # Add another mark while the mouse button is held down.
        if event.inaxes == self.ax:
            self.draw(event)

    def release(self, event):
        # Stop adding marks after the mouse button is released.
        self.drawing = False

    def draw(self, event, circular=True):
        if self.drawing and event.inaxes == self.ax:
            x, y = int(event.xdata), int(event.ydata)
            if not circular:
                # Simple square brush, retained as an alternative brush shape.
                self.canvas[max(0, y-6):min(256, y+6), max(0, x-6):min(256, x+6)]+=0.5
                self.img_plot.set_data(self.canvas)
                self.fig.canvas.draw_idle()
            else:
                # Build a soft circular brush so hand-drawn strokes look smooth.
                brush_tensor_crop_rule = np.s_[max(0, y-self.radius)-y+self.radius:min(256, y+self.radius)-y+self.radius, max(0, x-self.radius)-x+self.radius:min(256, x+self.radius)-x+self.radius]
                canvas_brush_crop = np.s_[max(0, y-self.radius):min(256, y+self.radius), max(0, x-self.radius):min(256, x+self.radius)]
                
                self.canvas[max(0, y-self.radius):min(256, y+self.radius), max(0, x-self.radius):min(256, x+self.radius)]+= self.soft_brush[brush_tensor_crop_rule]
                np.clip(self.canvas[canvas_brush_crop], 0.0, 1.0, out=self.canvas[canvas_brush_crop])
        
                
                    
                self.img_plot.set_data(self.canvas)
                self.fig.canvas.draw_idle()

def check_accuracy(model, loader, device):
    """Count correct predictions and collect the predicted class for each batch."""
    all_vals = []
    correct= 0
    total = 0
    for i, (x, y) in enumerate(loader):
        print(f'batch: {i}')
        x = x.to(device)
        y = y.to(device)

        outp = model(x)

        conf, vals = outp.max(1)

        correct += (vals == y).int().sum().item()

        vals = list(vals.to('cpu').numpy())

        total += len(vals)

        all_vals.append(vals)

    return correct, total, all_vals




if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('using apple\'s gpu!')
else:
    device = torch.device('cpu')
    print('unable to accelerate, using cpu')

image = Digitdrawer()
    
print(f'tensor has the shape: {image.canvas.shape}')

cleaned_image = prepare_for_model(image.canvas).to(device=device, dtype=torch.float32)
print(f' the shape of the cleaned image is: {cleaned_image.shape}')

plt.imshow(cleaned_image.squeeze().squeeze().to('cpu'), cmap='grey', vmin=0, vmax=1)
plt.title('cleaned image')
plt.axis('off')
plt.show()
plt.close()

# Defining the paths for the weights so that it can be loaded
script_dir = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(script_dir, "_weights"), exist_ok=True)
weights_folder = os.path.join(script_dir, '_weights')
weights_path = os.path.join(weights_folder, 'MNIST_w.pt')

classifier = pers_cls(1, 10, (3,2,1,1)).to(device)

if os.path.exists(weights_path):
    should_train = False
    classifier.load_state_dict(torch.load(weights_path))
else:
    print('Classifier model does not exist to train, exiting...')
    exit()

classifier.eval()
with torch.no_grad():
    one_hot_choice = classifier(cleaned_image)

    choice = torch.argmax(one_hot_choice)



    one_hot_choice = F.softmax(one_hot_choice.squeeze()*0.99, dim=0)

    print(one_hot_choice.shape)
    
print(f'the model predicted the number {choice} with the confidence of {one_hot_choice[choice]*100:.4f}')
response_to_draw = input('do you want to test again Y/n: ')
try:
    if response_to_draw.capitalize()== "Y":
        wanttodraw = True  
    elif response_to_draw.capitalize() == 'N':
        wanttodraw = False
except:
    print('input was invalid, exiting...')
    wanttodraw = False

