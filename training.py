import torch 
import torch.nn as nn
import torchvision
import numpy as np
from CNN import Net
from dataset import MotionDataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt 

def main():

    # FOR TUESDAY:
    # get more training data
    # try to figure out what kmp wants with the h.264 stuff and the entropy coding thing

    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #shuffle_dataset = true

    data_path = "train.csv"
    dataset = MotionDataset(data_path)

    num_epochs = 5
    num_classes = 10
    batch_size = 64
    learning_rate = 0.00001
    xCord = []
    xCord2 = []
    yCord = []
    yCord2 = []
    count = 0 
    count2 = 0

    MODEL_STORE_PATH = '/Users/edesern/Documents/COMP495/Video/videostream'

    softmax = nn.Softmax(dim=1)

    dataset_size = len(dataset)
    #print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))
    #if shuffle_dataset :
    #    np.random.seed(random_seed)
    #    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    #print(len(train_indices))
    #print(len(val_indices))


    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, 
                                                    sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                    sampler=valid_sampler)

    model = Net()
    model = model.float()
    #for param in model.parameters():
    #    param.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    #loss = 0

    for epoch in range(num_epochs):
        for i, sample in enumerate(train_loader):

            # takes sample data and goes into the neural network
            outputs = model(sample['input'].float())

            # splits the [100][12] array into two [100][6] arrays
            # pred0 and pred1 are the values after going into the neural network
            confi0 = outputs[:,0]
            confi0 = nn.Sigmoid()(confi0)
            # 2x3 : 0:6, 6:12
            pred0 = outputs[:,1:14]
            pred1 = outputs[:,14:27]

            # splits the [100][12] array into two [100][6] arrays
            # sol0 and sol1 are the ground truth values
            nomotion = sample['output'][:,0].float()
            sol0 = sample['output'][:,1:14].float()
            sol1 = sample['output'][:,14:27].float()

            print("ground truth motion, x and y")
            print(nomotion)
            print(sol0)
            print(sol1)


            # gives the index of the max value and turns pred0 and pred1 into size [100][1]
            #_, pred0 = pred0.max(1)
            #_, pred1 = pred1.max(1)

            _, sol0 = sol0.max(1)
            _, sol1 = sol1.max(1)

            #sol0 = sol0.long()
            #sol1 = sol1.long()

            loss = (nomotion - confi0).abs().sum()

            print("loss from nomotion prediction:", (nomotion - confi0).abs())


            mvpred0 = pred0.masked_select((1-nomotion.byte())[:, None]).reshape([-1,13])
            mvsol0 = sol0.masked_select(1-nomotion.byte())
            loss += criterion(mvpred0, mvsol0)
            print("loss of x criterion")
            print(criterion(pred0, sol0))

            mvpred1 = pred1.masked_select((1-nomotion.byte())[:, None]).reshape([-1,13])
            mvsol1 = sol1.masked_select(1-nomotion.byte())
            loss += criterion(mvpred1, mvsol1)
            print("loss of y criterion")
            print(criterion(pred1, sol1))

            #loss += criterion(pred0 * (1-nomotion)[:, None], sol0)
            #loss += criterion(confi0, nomotion0)
            #loss += criterion(confi1, nomotion1)
            #loss += criterion(pred1 * (1-nomotion)[:, None], sol1)
            loss_list.append(loss.item())

            

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = sample['output'].size(0)
            # together = [sol0, sol1]
            # print(together)
            # _, predicted = torch.max(outputs.data, 1)
            # exit()
            _, predicted0 = softmax(pred0).max(1)
            _, predicted1 = softmax(pred1).max(1)

            print("index of predicted x and y")

            print(predicted0 - 6)
            print(predicted1 - 6)

            print("raw outputs of motion, x and y")

            print(confi0)
            print(softmax(pred0))
            print(softmax(pred1))

            exit()


            # print(predicted0[0], sol0[0])
            # print(predicted1[5], sol1[5])



            # should be comparing sol0 and sol1 with predicted values
            # predicted should have 2 values.. so should be [100, 2]
            # correct = (predicted == sample['output']).sum().item()
            correct = (predicted0 == sol0).sum().item()
            correct += (predicted1 == sol1).sum().item()
            acc_list.append(correct / (total * 2))

            yCord.append(loss.item())
            count += 1
            xCord.append(count)
            yCord2.append((correct / (total * 2)) * 100)
            count2 += 1
            xCord2.append(count2)




            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / (total * 2)) * 100))
                

    plt.plot(xCord, yCord) 

  
    # naming the x axis 
    plt.xlabel('x - axis') 
    # naming the y axis 
    plt.ylabel('y - axis') 
  
    # giving a title to my graph 
    plt.title('Loss') 
  
    # function to show the plot 
    plt.show() 
    plt.plot(xCord2, yCord2)
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.title("Accuracy")
    plt.show()
            
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, sample in enumerate(test_loader):
            outputs = model(sample['input'].float())
            #confi0 = outputs[:,0]
            #confi1 = outputs[:,6]
            pred0 = outputs[:,1:14]
            pred1 = outputs[:,14:27]
            #nomotion0 = sample['output'][:,0]
            #nomotion1 = sample['output'][:,6]
            sol0 = sample['output'][:,1:14]
            sol1 = sample['output'][:,14:27]
            _, sol0 = sol0.max(1)
            _, sol1 = sol1.max(1)
            _, predicted0 = torch.max(pred0, 1)
            _, predicted1 = torch.max(pred1, 1)
            _, predicted0 = softmax(pred0).max(1)
            _, predicted1 = softmax(pred1).max(1)
            total = sample['output'].size(0)
            correct = (predicted0 == sol0).sum().item()
            correct += (predicted1 == sol1).sum().item()

        print('Test Accuracy of the model: {} %'.format((correct / (total * 2)) * 100))

    # Save the model and plot
    torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')


main()
