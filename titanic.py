import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

train = pd.read_csv('train.csv',encoding='utf-8')
test = pd.read_csv('test.csv',encoding='utf-8')
print('训练集：',train.shape)
print('测试集：',test.shape)

full = pd.concat([train,test],ignore_index=True)

#-----------------------------------------------------------------------------------------------
#特征工程

#对于价格一个的缺失值，这是一个S港的三等男性乘客，用S港三等乘客的票价中位数代替缺失值，票价替换为8.05。
full['Fare'] = full['Fare'].fillna(8.05)

#对于港口属性，由于缺失Embarked信息的乘客的Pclass均为1，且Fare均为80，而Pclass为1且Embarked为C的乘客的Fare中位数近似为80，所以将缺失值填充为C
full['Embarked'] = full['Embarked'].fillna('C')

#对于缺失值最多的Cabin属性,先填充为U,再转换为Deck属性
full['Cabin'] = full['Cabin'].fillna('U')
full['Deck'] = full['Cabin'].map(lambda x: x[0])
deckDF = pd.get_dummies(full['Deck'],prefix='Cabin')


#对性别属性处理，把男女分别映射为数字 1，0
sex_mapDict = {'male':1,'female':0}
full['Sex'] = full['Sex'].map(sex_mapDict)


#对特征使用DataFrame二维标签结构存储，使用独热编码对离散特征进行处理
embarkedDF = pd.get_dummies(full['Embarked'],prefix='Embarked')

#对于ticket属性处理:增加TicketGroup属性，因为同票号乘客对存活率有影响
Ticket_Count = dict(full['Ticket'].value_counts())
full['TicketGroup'] = full['Ticket'].apply(lambda x:Ticket_Count[x])

def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0
full['TicketGroup'] = full['TicketGroup'].apply(Ticket_Label)


#处理家庭属性
familyDF = pd.DataFrame()
#家庭人数=父母子女+兄弟姐妹+自己
familyDF['FamilySize'] = full['SibSp']+full['Parch']+1
#自定义家庭类别
familyDF['Family_Single'] = familyDF['FamilySize'].map(lambda s: 1 if ((2<=s<5)|(s==1)) else 0)
familyDF['Family_Small'] = familyDF['FamilySize'].map(lambda s: 1 if 5<=s<=7 else 0)
familyDF['Family_Large'] = familyDF['FamilySize'].map(lambda s: 1 if s>7 else 0)

#Name中的头衔信息title
#姓名中逗号前面的是'名'，逗号后面是'头衔.姓'
full['Title'] = full.Name.map(lambda x:x.split(',')[1].split('.')[0].strip())  #strip()用于移除字符串头尾指定的字符（默认为空格）
#姓名中头衔字符串与定义头衔类别的映射关系
title_Dict={
    'Capt':           'Officer', #政府官员
    'Col':            'Officer',
    'Don':            'Royalty', #王室
    'Dona':           'Royalty',
    'Dr':             'Officer',
    'Jonkheer':       'Master', #专家
    'Lady':           'Royalty',
    'Major':          'Officer',
    'Master':         'Master',
    'Miss':           'Miss', #未婚女子
    'Mlle':           'Miss',
    'Mme':            'Mrs', #已婚男子
    'Mr':             'Mr', #未婚男子
    'Mrs':            'Mrs',
    'Ms':             'Mrs',
    'Rev':            'Officer',
    'Sir':            'Royalty',
    'the Countess':   'Royalty'
}
full['Title'] = full['Title'].map(title_Dict)
titleDF = pd.get_dummies(full['Title'],prefix='Title')


#对于Age属性，通过用Sex, Title, Pclass三个特征构建随机森林模型，填充年龄缺失值

ageDF = full[['Age', 'Pclass','Sex','Title']]
ageDF = pd.get_dummies(ageDF)
known_age = ageDF[ageDF.Age.notnull()].values
unknown_age = ageDF[ageDF.Age.isnull()].values
y = known_age[:, 0]
X = known_age[:, 1:]
rfr = RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
rfr.fit(X, y.astype(int))
predictedAges = rfr.predict(unknown_age[:, 1::])
full.loc[ (full.Age.isnull()), 'Age' ] = predictedAges



#用embarked同样的方法进行Pclass处理
pclassDF = pd.get_dummies(full['Pclass'],prefix='Pclass')


#整合数据
full = pd.concat([full,pclassDF,titleDF,familyDF,deckDF,embarkedDF],axis=1)
full.drop(['Pclass','Title','Embarked','Deck'],axis=1,inplace=True)



'''
#--------------------------------------------------------------------------

异常值处理：（参考kaggle高分大佬的经验，但是在我实际测试中得出的准确率并没有明显提高，故并未使用）
把姓氏相同的乘客划分为同一组，从人数大于1的组中分别提取出每组的妇女儿童和成年男性。

full['Surname'] = full['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(full['Surname'].value_counts())
full['FamilyGroup'] = full['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group = full.loc[(full['FamilyGroup']>=2) & ((full['Age']<=12) | (full['Sex']=='female'))]
Male_Adult_Group = full.loc[(full['FamilyGroup']>=2) & (full['Age']>12) & (full['Sex']=='male')]
#发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难。
Female_Child = pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns = ['GroupCount']

#绝大部分成年男性组的平均存活率也为1或0。
Male_Adult = pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns = ['GroupCount']


普遍规律是女性和儿童幸存率高，成年男性幸存较低，所以我们把不符合普遍规律的反常组选出来单独处理。
把女性和儿童幸存率为0的组设置为遇难组，把成年男性存活率为1的组设置为幸存组。

Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)

Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)

#为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
train = full.loc[full['Survived'].notnull()]
test = full.loc[full['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 1
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 0
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'


titleDF = pd.get_dummies(full['Title'],prefix='Title')


#重新划分数据，分出训练集和测试集
full = pd.concat([train, test])

#重新整合数据
full = pd.concat([full,pclassDF,titleDF,familyDF,deckDF,embarkedDF],axis=1)


#-------------------------------------------------------------------------------

'''

#-------------------------------------------------------------------------------




#特征提取
#根据和存活度的正相关性选取特征
full_X=pd.concat([full['Age'],
                  deckDF,
                  titleDF,
                  pclassDF,
                  familyDF,
                  #full['Fare'], #和存活度相关性较差
                  #embarkedDF, #和存活度相关性较差
                  full['Sex'],
                  full['TicketGroup']
                  ],axis = 1)

#原始数据美有891行
sourceRow = 891
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']
#预测数据集：特征
pred_X=full_X.loc[sourceRow:,:]

# 建立模型用的训练数据美和测试数据美
train_X, test_X, train_y, test_y = train_test_split(source_X, source_y, train_size=0.8)
'''
# 打印测试数据大小
print('原始数据集特征:', source_X.shape,
      '训练数据集特征:', train_X.shape,
      '测试数据集特征:', test_X.shape)

print('原始数据集标签:', source_y.shape,
      '训练数据集标签:', train_y.shape,
      '测试数据集标签:', test_y.shape)
'''
print('--------------------------------------\n')#分割线


#汇总不同模型算法-----------------------------------------

#1.使用K临近算法模型
knc_model = KNeighborsClassifier(n_neighbors = 3)
knc_model.fit(train_X,train_y)


#2.使用GDBT梯度提升树模型
gbc_model = GradientBoostingClassifier()
gbc_model.fit(train_X,train_y)


#3.使用极度随机树模型
etc_model = ExtraTreesClassifier(random_state = 6, bootstrap=True, oob_score=True)
etc_model.fit(train_X,train_y)


#4.使用决策树模型
dtc_model = DecisionTreeClassifier()
dtc_model.fit(train_X,train_y)


#5.使用逻辑回归模型
lr_model = LogisticRegression(penalty="l2",solver="liblinear",C=0.9850000000000004,max_iter=5000)
lr_model.fit(train_X,train_y)


#6,使用支持向量机svm模型
svm_model = SVC(C=10,kernel='rbf',gamma=0.05,decision_function_shape='ovr')
svm_model.fit(train_X,train_y)


#7.使用随机森林模型:目前测试最好
rfc_model=RandomForestClassifier(random_state = 0,
                                 min_samples_split = 8,
                                 n_estimators = 180,
                                 min_samples_leaf=2,
                                 max_depth = 8)
rfc_model.fit(train_X,train_y)

#深度学习---------------------------------------------------------------------------
#8.使用自定义的深度学习网络模型，使用梯度下降
'''
使用torch.from_numpy更加安全，使用tensor.Tensor在非float类型下会与预期不符。
'''
dl_train_X = torch.from_numpy(np.array(train_X))  # 将数据转化为tensor格式
dl_train_y = torch.from_numpy(np.array(train_y))
dl_test_X = torch.from_numpy(np.array(test_X))
dl_test_y = torch.from_numpy(np.array(test_y))
in_features = dl_train_X.shape[1]

#print(in_features)作为神经网络模型的输入特征数量


# 构造网络模型
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear1 = torch.nn.Linear(25, 64)#加上注释特征，输入为29
        self.linear2 = torch.nn.Linear(64, 10)
        self.linear3 = torch.nn.Linear(10, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

'''
#改良版的Net,效果一般
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear1 = nn.Linear(25, 64)#加上注释特征，输入为29
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.dropout(self.linear2(x)))
        x = self.relu(self.dropout(self.linear3(x)))
        x = self.relu(self.linear4(x))
        x = self.softmax(self.linear5(x))
        return x
'''

dl_model = model()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.01)

epochs = 1000

# 开始训练
for epoch in range(epochs):
    # 正向传播
    y_pred = dl_model(dl_train_X.float())
    loss = criterion(torch.squeeze(y_pred), dl_train_y.float())
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    # 更新参数
    optimizer.step()



#---------------------------------------------------------------------------------


#k折交叉验证
def cross_score(X,y,mod):
     #简单看看打分情况
    scores=cross_val_score(mod,X,y,cv=10)
    print('交叉验证scores:',scores)
    print('交叉验证mean:',scores.mean())
    print('--------------------------------------\n')
    return scores.mean()


models = [knc_model,gbc_model,etc_model,dtc_model,lr_model,svm_model,rfc_model]
model_names = ['knc_model','gbc_model','etc_model','dtc_model','lr_model','svm_model','rfc_model']



def models_score(source_X,source_y,models,model_names,train_X, train_y,test_X, test_y):
    best_score = cross_val_score(models[0],source_X, source_y, cv=10).mean()
    best_score_name = model_names[0]
    i = 0
    best_model = models[0]
    for model in models:
        print(model_names[i], "train score:", model.score(train_X, train_y))
        print(model_names[i], "test score:", model.score(test_X, test_y))
        score = cross_score(source_X, source_y, model)
        if(score > best_score):
            best_score_name = model_names[i]
            best_score = score
            best_model = model
        i = i + 1
    print('the best score model is',best_score_name,',cross score is',best_score)
    return best_model

model = models_score(source_X,source_y,models,model_names,train_X, train_y,test_X, test_y)




'''
#提交模块：预测值为浮点型，需要转化整型提交
#1。sklearn模型提交
pred_Y = model.predict(pred_X)
pred_Y = pred_Y.astype(int)


#---------------------------------------------------------------

#2。自定义deep learning模型提交
pred_X = torch.from_numpy(np.array(pred_X))#转为tensor的形式
pred_Y = dl_model(pred_X.float())
pred_Y = pred_Y.detach().numpy()#将tensor转为ndarry形式
pred_Y = pred_Y.astype(int)#转为整型提交
'''

'''
#生成提交文件.csv
passenger_id = full.loc[sourceRow:,'PassengerId']

predDF = pd.DataFrame(
{
 'PassengerId':passenger_id,
 'Survived' :pred_Y
})

predDF.to_csv('zxy_titanic_pred.csv',index=False)
print('run successful')
'''

