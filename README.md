We use following block of code to download MNIST data as Tensor.
    test_dataset = datasets.MNIST(
    root='.',
    train=False,
    transform=transforms.ToTensor(),
    download=True
    )
Downloaded data is in shape :
    torch.Size([10000, 28, 28])

For data generation we have used following code block:
class customMNISTDataset(Dataset):

  def __init__(self,dat):
      self.X = dat.data
      self.Y = dat.targets
  
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self,idx):
    img_data = self.X[idx]
    rand_no = randint(0,9)
    y1 = self.Y[idx]
    y2 = y1+rand_no
    return img_data, rand_no, y1,y2
    
so data is generated as group of image data, random number, target label, target sum

To combine the two inputs, I have taken the output of image data after fc3 layer and performed Softmax then argmax on this data.
Then computed one hot of this data and added to the already present one hot of the random numbers which was input to the network.
 It can be seen in following code :
     t1 = F.softmax(t1,dim=1)
     t_out = t1.argmax(dim=1)
     t_out = F.one_hot(torch.tensor(t_out), num_classes=10).float()
     rand_num = t_out.add(rand_num)

As of now not satisfied with the training only, I can see that network is giviong same error in each epoch.
The result currently is total loss for labels and total loss for sum.

Picked up loss function as cross entropy for categorical issue of image identification, and loss for sum is calculated as Mean square error as considering this problem as regression problem.
