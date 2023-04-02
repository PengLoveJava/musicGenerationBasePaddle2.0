from Transform.Transform import *
import random
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

def dataLoader(path='./dataset'):
    BATCHSIZE = 32
    notes = createNotes(path)
    inputs = createSeq(notes)

    datas = np.array(inputs)
    index_list = list(range(len(datas)))

    def dataGenerator():
        random.shuffle(index_list)
        datas_list = []
        for i in index_list:
            data = datas[i].astype('float32')
            datas_list.append(data)
            if len(datas_list) == BATCHSIZE:
                yield np.array(datas_list)
                datas_list = []
        if len(datas_list) > 0:
            yield np.array(datas_list)

    return dataGenerator

class D(paddle.nn.Layer):
    def __init__(self, name_scope='D_'):
        super(D, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=1,
            hidden_size=256,
            num_layers=3
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=512,
            direction='bidirectional'
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm(512),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm(256),
            nn.LeakyReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )


    def forward(self, inputs):
        x, (_, _) = self.lstm1(inputs)
        x, (_, _) = self.lstm2(x)
        x = x[:,-1,:]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

Discriminator = D()
paddle.summary(Discriminator, (8, 100, 1))

class G(paddle.nn.Layer):
    def __init__(self, name_scope='G_'):
        super(G, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm(256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 100),
            nn.Tanh()
        )
    def forward(self, inputs):
        x = self.generator(inputs)
        x = paddle.unsqueeze(x, axis=-1)
        return x

Generator = G()
paddle.summary(Generator, (8, 100))

def train():
    train_loader = dataLoader()
    Discriminator = D()
    Generator = G()

    Discriminator.train()
    Generator.train()

    optim1 = paddle.optimizer.Adam(parameters=Discriminator.parameters(), weight_decay=0.001, learning_rate=1e-5)
    optim2 = paddle.optimizer.Adam(parameters=Generator.parameters(), weight_decay=0.001, learning_rate=1e-5)

    epoch_num = 10
    train_g = 2
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_loader()):
            real = paddle.to_tensor(data)
            noise = paddle.uniform((real.shape[0], 100))
            fake = Generator(noise)

            fake_loss = F.binary_cross_entropy(Discriminator(fake), paddle.zeros((real.shape[0], 1)))
            real_loss = -1 * F.binary_cross_entropy(Discriminator(real), paddle.ones((real.shape[0], 1)))

            d_loss = fake_loss + real_loss

            d_loss.backward()
            optim1.step()
            optim1.clear_grad()

            for _ in range(train_g):
                noise = paddle.uniform((real.shape[0], 100))
                fake = Generator(noise)
                g_loss = -1 * F.binary_cross_entropy(Discriminator(fake), paddle.zeros((real.shape[0], 1)))

                g_loss.backward()
                optim2.step()
                optim2.clear_grad()

            if batch_id % 64 == 0:
                print("epoch: {}, batch_id: {}, d_loss is: {}, g_loss is: {}".format(epoch, batch_id, d_loss.numpy(),
                                                                                     g_loss.numpy()))

        if epoch % 2 == 0:
            noise = paddle.uniform((real.shape[0], 100))
            fake = Generator(noise)
            musicGenerator(fake.numpy())

        if epoch % 2 == 0:
            paddle.save(Discriminator.state_dict(), './model/D.pdparams')
            paddle.save(optim1.state_dict(), './model/D.pdopt')
            paddle.save(Generator.state_dict(), './model/G.pdparams')
            paddle.save(optim2.state_dict(), './model/G.pdopt')


if __name__ == '__main__':
    train()