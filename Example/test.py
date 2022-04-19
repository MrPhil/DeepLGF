import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, Lambda, GaussianNoise, BatchNormalization, Reshape, dot, Activation, concatenate, AveragePooling1D, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.engine.topology import Layer
from keras.utils import plot_model
from keras.datasets import mnist
from keras import backend as K
from random import shuffle
from keras.callbacks import ReduceLROnPlateau
import csv
import numpy as np
import datetime
import tensorflow as tf
# 定义函数
def ReadMyCsv1(SaveList, fileName):  # 按行读 并转化为list，大list每个元素是小list(名字与结构组成)，row是小list
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return
def GenerateBehaviorFeature(InteractionPair, NodeBehavior, net):
    SampleFeature1 = []
    SampleFeature2 = []
    i = 0
    while i < (len(InteractionPair)):
        print(i)
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]
        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:
                SampleFeature1.append(NodeBehavior[m][1:])
                break
            if m == (len(NodeBehavior)-1):
                for p in range(len(net)):
                    flag = 0
                    if Pair1 == net[p][0]:
                        Pair11 = net[p][1]
                        for x in range(len(NodeBehavior)):
                            if Pair11 == NodeBehavior[x][0]:
                                # a = suiji  NodeBehavior[x][1:]
                                a = NodeBehavior[x][1:]
                                SampleFeature1.append(a)
                                flag = 1
                                break
                    if flag == 1:
                        break
                    if Pair1 == net[p][1]:
                        Pair12 = net[p][0]
                        for x in range(len(NodeBehavior)):
                            if Pair12 == NodeBehavior[x][0]:
                                # a = suiji  NodeBehavior[x][1:]
                                a = NodeBehavior[x][1:]
                                SampleFeature1.append(a)
                                flag = 1
                                break
                    if flag == 1:
                        break
                    if p == len(net)-1:
                        b = np.zeros(len(NodeBehavior[x][1:]))
                        SampleFeature1.append(b)
        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:
                SampleFeature2.append(NodeBehavior[n][1:])
                break
            if n == (len(NodeBehavior)-1):
                if n == (len(NodeBehavior) - 1):
                    for p in range(len(net)):
                        flag = 0
                        if Pair2 == net[p][0]:
                            Pair21 = net[p][1]
                            for x in range(len(NodeBehavior)):
                                if Pair21 == NodeBehavior[x][0]:
                                    a = NodeBehavior[x][1:]
                                    SampleFeature2.append(a)
                                    flag = 1
                                    break
                        if flag == 1:
                            break
                        if Pair2 == net[p][1]:
                            Pair22 = net[p][0]
                            for x in range(len(NodeBehavior)):
                                if Pair22 == NodeBehavior[x][0]:
                                    a = NodeBehavior[x][1:]
                                    SampleFeature2.append(a)
                                    flag = 1
                                    break
                        if flag == 1:
                            break
                        if p == len(net) - 1:
                            b = np.zeros(len(NodeBehavior[x][1:]))
                            SampleFeature2.append(b)
        i = i + 1
    return SampleFeature1, SampleFeature2

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label

if __name__ == '__main__':

    nameTModel = 'direct_f'

    nameF = '0/'

    representationFile = 'representation/'

    AllNodeStructure = []
    ReadMyCsv1(AllNodeStructure, representationFile + 'index-SMILES1940-vector.csv')  # 0: 1898*
    AllNodeSimilarity = []
    ReadMyCsv1(AllNodeSimilarity, representationFile + 'drug_representations.csv')  # 0: 1898*
    AllNodeKG = []
    ReadMyCsv1(AllNodeKG, representationFile + 'drug_KG.csv')  # 0: 1898*

    PositiveSample_Train = []
    ReadMyCsv1(PositiveSample_Train, nameF + 'rPositiveSample_Train_ddi.csv')  # 153472
    PositiveSample_Validation = []
    ReadMyCsv1(PositiveSample_Validation, nameF + 'rPositiveSample_Validation_ddi.csv')  # 21925
    PositiveSample_Test = []
    ReadMyCsv1(PositiveSample_Test, nameF + 'rPositiveSample_Test_ddi.csv')  # 43850

    NegativeSample_Train = []
    ReadMyCsv1(NegativeSample_Train, nameF + 'rNegativeSample_Train_ddi.csv')
    NegativeSample_Validation = []
    ReadMyCsv1(NegativeSample_Validation, nameF + 'rNegativeSample_Validation_ddi.csv')
    NegativeSample_Test = []
    ReadMyCsv1(NegativeSample_Test, nameF + 'rNegativeSample_Test_ddi.csv')

    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)
    x_train_pair.extend(NegativeSample_Train)
    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)
    x_test_pair = []
    x_test_pair.extend(PositiveSample_Test)
    x_test_pair.extend(NegativeSample_Test)


    DDI_net = []
    ReadMyCsv1(DDI_net, 'drugInteractionNet.csv')
    x_train_1_Structure, x_train_2_Structure = GenerateBehaviorFeature(x_train_pair, AllNodeStructure, DDI_net)
    np.save(nameF + "x_train_1_Structure.npy", x_train_1_Structure)
    x_train_1_Structure = np.load(nameF + "x_train_1_Structure.npy", allow_pickle=True)
    np.save(nameF + "x_train_2_Structure.npy", x_train_2_Structure)
    x_train_2_Structure = np.load(nameF + "x_train_2_Structure.npy", allow_pickle=True)
    x_validation_1_Structure, x_validation_2_Structure = GenerateBehaviorFeature(x_validation_pair, AllNodeStructure, DDI_net)
    np.save(nameF + "x_validation_1_Structure.npy", x_validation_1_Structure)
    x_validation_1_Structure = np.load(nameF + "x_validation_1_Structure.npy", allow_pickle=True)
    np.save(nameF + "x_validation_2_Structure.npy", x_validation_2_Structure)
    x_validation_2_Structure = np.load(nameF + "x_validation_2_Structure.npy", allow_pickle=True)
    x_test_1_Structure, x_test_2_Structure = GenerateBehaviorFeature(x_test_pair, AllNodeStructure, DDI_net)
    np.save(nameF + "x_test_1_Structure.npy", x_test_1_Structure)
    x_test_1_Structure = np.load(nameF + "x_test_1_Structure.npy", allow_pickle=True)
    np.save(nameF + "x_test_2_Structure.npy", x_test_2_Structure)
    x_test_2_Structure = np.load(nameF + "x_test_2_Structure.npy", allow_pickle=True)

    x_train_1_Similarity, x_train_2_Similarity = GenerateBehaviorFeature(x_train_pair, AllNodeSimilarity, DDI_net)
    np.save(nameF + "x_train_1_Similarity.npy", x_train_1_Similarity)
    x_train_1_Similarity = np.load(nameF + "x_train_1_Similarity.npy", allow_pickle=True)
    np.save(nameF + "x_train_2_Similarity.npy", x_train_2_Similarity)
    x_train_2_Similarity = np.load(nameF + "x_train_2_Similarity.npy", allow_pickle=True)
    x_validation_1_Similarity, x_validation_2_Similarity = GenerateBehaviorFeature(x_validation_pair, AllNodeSimilarity, DDI_net)
    np.save(nameF + "x_validation_1_Similarity.npy", x_validation_1_Similarity)
    x_validation_1_Similarity = np.load(nameF + "x_validation_1_Similarity.npy", allow_pickle=True)
    np.save(nameF + "x_validation_2_Similarity.npy", x_validation_2_Similarity)
    x_validation_2_Similarity = np.load(nameF + "x_validation_2_Similarity.npy", allow_pickle=True)
    x_test_1_Similarity, x_test_2_Similarity = GenerateBehaviorFeature(x_test_pair, AllNodeSimilarity, DDI_net)
    np.save(nameF + "x_test_1_Similarity.npy", x_test_1_Similarity)
    x_test_1_Similarity = np.load(nameF + "x_test_1_Similarity.npy", allow_pickle=True)
    np.save(nameF + "x_test_2_Similarity.npy", x_test_2_Similarity)
    x_test_2_Similarity = np.load(nameF + "x_test_2_Similarity.npy", allow_pickle=True)

    x_train_1_KG, x_train_2_KG = GenerateBehaviorFeature(x_train_pair, AllNodeKG, DDI_net)
    np.save(nameF + "x_train_1_KG.npy", x_train_1_KG)
    x_train_1_KG = np.load(nameF + "x_train_1_KG.npy", allow_pickle=True)
    np.save(nameF + "x_train_2_KG.npy", x_train_2_KG)
    x_train_2_KG = np.load(nameF + "x_train_2_KG.npy", allow_pickle=True)
    x_validation_1_KG, x_validation_2_KG = GenerateBehaviorFeature(x_validation_pair, AllNodeKG, DDI_net)
    np.save(nameF + "x_validation_1_KG.npy", x_validation_1_KG)
    x_validation_1_KG = np.load(nameF + "x_validation_1_KG.npy", allow_pickle=True)
    np.save(nameF + "x_validation_2_KG.npy", x_validation_2_KG)
    x_validation_2_KG = np.load(nameF + "x_validation_2_KG.npy", allow_pickle=True)
    x_test_1_KG, x_test_2_KG = GenerateBehaviorFeature(x_test_pair, AllNodeKG, DDI_net)
    np.save(nameF + "x_test_1_KG.npy", x_test_1_KG)
    x_test_1_KG = np.load(nameF + "x_test_1_KG.npy", allow_pickle=True)
    np.save(nameF + "x_test_2_KG.npy", x_test_2_KG)
    x_test_2_KG = np.load(nameF + "x_test_2_KG.npy", allow_pickle=True)

    y_train_Pre = MyLabel(x_train_pair)  # Label->one hot
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)
    num_classes = 2
    y_train = keras.utils.to_categorical(y_train_Pre, num_classes)
    y_validation = keras.utils.to_categorical(y_validation_Pre, num_classes)
    y_test = keras.utils.to_categorical(y_test_Pre, num_classes)

    print('x_train_1_Structure shape', x_train_1_Structure.shape)
    print('x_train_2_Structure shape', x_train_2_Structure.shape)
    print('x_train_1_Similarity shape', x_train_1_Similarity.shape)
    print('x_train_2_Similarity shape', x_train_2_Similarity.shape)
    print('x_train_1_KG shape', x_train_1_KG.shape)
    print('x_train_2_KG shape', x_train_2_KG.shape)

    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)


    # ———————————————————— define ————————————————————
    # # ——输入1
    input1 = Input(shape=(len(x_train_1_Structure[0]),), name='input1')
    x1 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(input1)
    x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x1)
    x1 = Dropout(rate=0.3)(x1)
    # ——输入2——
    input2 = Input(shape=(len(x_train_2_Structure[0]),), name='input2')
    x2 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(input2)
    x2 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x2)
    x2 = Dropout(rate=0.3)(x2)
    # ——输入3——
    input3 = Input(shape=(len(x_train_1_Similarity[0]),), name='input3')
    x3 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(input3)
    x3 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x3)
    x3 = Dropout(rate=0.3)(x3)
    # ——输入4——
    input4 = Input(shape=(len(x_train_2_Similarity[0]),), name='input4')
    x4 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(input4)
    x4 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x4)
    x4 = Dropout(rate=0.3)(x4)
    # ——输入5——
    input5 = Input(shape=(len(x_train_1_KG[0]),), name='input5')
    x5 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(input5)
    x5 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x5)
    x5 = Dropout(rate=0.3)(x5)
    # ——输入6——
    input6 = Input(shape=(len(x_train_2_KG[0]),), name='input6')
    x6 = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.001))(input6)
    x6 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.001))(x6)
    x6 = Dropout(rate=0.3)(x6)

    if nameTModel == 'direct_f':
        flatten = keras.layers.concatenate([x1, x2, x3, x4, x5, x6])

    # ——全连接——
    hidden = Dense(256, activation='relu', name='hidden1', activity_regularizer=regularizers.l2(0.001))(flatten)
    hidden = Dropout(rate=0.3)(hidden)
    hidden = Dense(128, activation='relu', name='hidden3', activity_regularizer=regularizers.l2(0.001))(hidden)
    hidden = Dropout(rate=0.3)(hidden)
    hidden = Dense(32, activation='relu', name='hidden4', activity_regularizer=regularizers.l2(0.001))(hidden)
    hidden = Dropout(rate=0.3)(hidden)
    output = Dense(num_classes, activation='softmax', name='output')(hidden)  # category

    model = Model(inputs=[input1, input2, input3, input4, input5, input6], outputs=output)
    # 打印网络结构
    model.summary()
    # ——编译——
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # —————————————————————— train ——————————————————————
    history = model.fit({'input1': x_train_1_Structure, 'input2': x_train_2_Structure, 'input3': x_train_1_Similarity, 'input4': x_train_2_Similarity,
                         'input5': x_train_1_KG, 'input6': x_train_2_KG}, y_train,
                        validation_data=({'input1': x_validation_1_Structure, 'input2': x_validation_2_Structure,
                                          'input3': x_validation_1_Similarity, 'input4': x_validation_2_Similarity,
                                          'input5': x_validation_1_KG, 'input6': x_validation_2_KG}, y_validation),
                        epochs=200, batch_size=128,
                        )

    ModelTest = Model(inputs=model.input, outputs=model.get_layer('output').output)
    ModelTestOutput = ModelTest.predict(
        [x_test_1_Structure, x_test_2_Structure, x_test_1_Similarity, x_test_2_Similarity, x_test_1_KG, x_test_2_KG])

    print(ModelTestOutput[:][1])

endtime = datetime.datetime.now()

