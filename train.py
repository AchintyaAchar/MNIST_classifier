import ssl

import time as t
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time as t
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
import matplotlib as mpl

# Prefer a GUI backend when one is available, but fall back cleanly in
# headless environments where Qt is not installed.
for backend in ('MacOSX', 'QtAgg', 'TkAgg', 'GTK3Agg', 'Agg'):
    try:
        mpl.use(backend, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import os
from Classifier import Conv_network as pers_cls






class noise(nn.Module):
    """Add Gaussian noise to activations while the model is training.

    Noise makes the network less dependent on exact training examples, which
    can improve generalization. The layer behaves like an identity layer in
    evaluation mode unless ``want_noise`` is explicitly enabled.
    """

    def __init__(self, noiseval=0.1):
        super().__init__()
        # Store the standard deviation used for the random noise.

        self.sdg = noiseval

        # ``self.training`` is toggled by model.train() and model.eval().
    def forward(self, x, want_noise=False):
        if self.training or want_noise:
            x = x+self.sdg*torch.randn_like(x)
        return x
    


def train(EPOCHS, report_after_epochs, traindata_loader, testdata_loader,
          device, classifier, optimizer, criterion, save_path):
    """Train the classifier, periodically evaluate it, and save its weights."""

    start_t = t.time()
    ep_start_t = t.time()
    running_loss = None
    for i in range(EPOCHS):
        # Each batch performs one complete forward/backward optimization step.
        for train in traindata_loader:
            image, label = train
            image = image.to(device)
            label = label.to(device)
            pred = classifier(image)
            # CrossEntropyLoss expects raw class scores and integer labels.
            train_loss = criterion(pred, label)
            # Accumulate loss so it can be reported as a batch average.
            if running_loss == None:
                running_loss = torch.zeros_like(train_loss)
                running_loss += train_loss
            else:
                running_loss += train_loss
            # Clear gradients from the previous batch before backpropagation.
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # Evaluate after the requested interval (including the first epoch).
        if i % report_after_epochs == 0:
            # Disable training-only behavior such as activation noise.
            classifier.eval()
            dur = t.time() - ep_start_t
            avg_train_loss = (running_loss / len(traindata_loader)).item()
            running_test_loss = None
            # Evaluation does not need gradients, saving memory and time.
            with torch.inference_mode():
                for image, label in testdata_loader:
                    image = image.to(device)
                    label = label.to(device)
                    pred = classifier(image)
                    test_loss = criterion(pred, label)
                    if running_test_loss == None:
                        running_test_loss = torch.zeros_like(test_loss)
                        running_test_loss += test_loss
                    else:
                        running_test_loss += test_loss

            avg_test_loss = (running_test_loss / len(testdata_loader)).item()
            print(f'train error is {avg_train_loss:.4f} at epoch {i} ')
            print(f'test error is {avg_test_loss:.4f}')
            if i > 9:
                print(f'each epoch took {dur / report_after_epochs:.4f} seconds \n')
            ep_start_t = t.time()
            classifier.train()

    duration = t.time() - start_t
    torch.save(classifier.state_dict(), save_path)
    return duration

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


        
        


def main():

    ssl._create_default_https_context = ssl._create_unverified_context

    #define all the file directories:

    

    script_dir = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(script_dir, "_weights"), exist_ok=True)
    os.makedirs(os.path.join(script_dir,"dataset_objs"), exist_ok=True)
    weights_folder = os.path.join(script_dir, '_weights')

    weights_path = os.path.join(weights_folder, 'MNIST_w.pt')

    dataset_paths = os.path.join(script_dir, 'dataset_objs')
    train_dataset_path = os.path.join(dataset_paths, "train_dataset.pt")
    test_dataset_path = os.path.join(dataset_paths, "test_dataset.pt")

    if not(os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path)):
        test_raw = MNIST(root='data/', download=True, transform=transforms.ToTensor(), train=False)
        train_raw = MNIST(root='data/', download=True, transform=transforms.ToTensor(), train=True)

        x_train = train_raw.data.float()/255.0
        x_train = x_train.unsqueeze(1)
        y_train = train_raw.targets

        x_test = test_raw.data.float()/255.0
        x_test = x_test.unsqueeze(1)
        y_test = test_raw.targets
    
        torch.save({'x':x_train, 'y':y_train}, train_dataset_path)
        torch.save({'x':x_test, 'y':y_test}, test_dataset_path)

        train_payload = torch.load(train_dataset_path, weights_only=True)
        test_payload = torch.load(test_dataset_path, weights_only=True)

        train_dataset = TensorDataset(train_payload['x'], train_payload['y'])
        test_dataset = TensorDataset(test_payload['x'], test_payload['y'])
        
        

    else:

        train_payload = torch.load(train_dataset_path, weights_only=True)
        test_payload = torch.load(test_dataset_path, weights_only=True)

        train_dataset = TensorDataset(train_payload['x'], train_payload['y'])
        test_dataset = TensorDataset(test_payload['x'], test_payload['y'])
        
    
            


    should_train = True

    traindata_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=4, persistent_workers=True)
    testdata_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4, persistent_workers=True)

    print(f'this is the length of the train dataset: {len(train_dataset)}')

    def test_dataset_image():
        image, label = test_dataset[np.random.randint(1,122)]
        print('this is a test image in the dataset')
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'label= {label}')
        plt.axis('off')
        plt.show()
        plt.close()

    #tests whether the dataset images are properly loading. Commented for now as not necessary

    #test_dataset_image()
 
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print('using apple\'s gpu!')
    else:
        device = torch.device('cpu')
        print('unable to accelerate, using cpu')

    

    
    classifier = pers_cls(1, 10, (3,2,1,1)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(lr=0.001, params=classifier.parameters(), weight_decay=1e-4)
    
    EPOCHS = 100
    report_after_epochs = 5

    if os.path.exists(weights_path):
        should_train = False
        try:
            classifier.load_state_dict(torch.load(weights_path))
            print('found pre-exsisting model saved, delete')
        except:
            print('The model did not load properly, possibly due to a change in the model architecture, exiting program anyways...')
            raise
            #exit()

    
    if should_train:

        duration = train(
            EPOCHS, 
            report_after_epochs, 
            traindata_loader, 
            testdata_loader, 
            device, 
            classifier, 
            optimizer, 
            criterion,
            weights_path
            )
    
    
    
    
    if should_train:
        print('TRAINING IS OVER!!')
        print(f'it took {duration:.4f} seconds to train model')

    print(f'the model has a parameter count of: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}')
    print('testing model')


    print('Finding accuracy of models...')

    correct, total, all_vals = check_accuracy(classifier, testdata_loader, device)

    all_vals = all_vals[0]

    print(f'The model got {correct} values correct out of {total} questions with an accuracy of {correct/total:.4f}\nA sample of the results are: {all_vals[:10]}')
  
    
if __name__ == '__main__':
    main()
    