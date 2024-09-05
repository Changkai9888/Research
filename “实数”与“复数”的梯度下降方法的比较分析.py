import torch,fc
'''
研究问题：
以一个函数为例，对“实数”与“复数”的梯度下降方法的比较分析。
为了探索，是否可用“复数域+平方根”的处理方式，代替通用的relu激活函数？

研究方法：
以pytorch为环境，以一个函数y=-(x+3)**0.5+10*(x-1)**0.5为例，
在x=0 为初始值，分别用实数值和复数值进行梯度下降法求极大值和极小值。

说明：目标函数图像：y=-(x+3)**0.5+10*(x-1)**0.5，对于实数方法，由于要开根号，做了relu处理，即：
y=-relu(x+3)**0.5+10*relu(x-1)**0.5

注：fc.plot函数附在最后
'''
####原始的函数图像：y=-(x+3)**0.5+10*(x-1)**0.5
y=[]
for i in range(1000):
    x=i*0.01-5
    y+=[(-(x+3)**0.5+10*(x-1)**0.5).real]
print('理论min为',min(y))
print('理论min在 x=',y.index(min(y))*0.01-5)
print('理论max为',max(y) ,'（-->正无穷）')
print('理论max在 x=',y.index(max(y))*0.01-5, '（-->正无穷）')
fc.plot(y,plot_name='原始图像y=Value；x轴：x=Index*0.01-5')
####分别用实分析和复分析梯度下降
def ansys(max_or_min,lr):
    #求y的最大值还是最小值？1：求最大值，-1，求最小值
    if max_or_min not in [1,-1]:
        print('参数有误！')
        return 
    str_='最大值' if max_or_min==1 else '最小值' if max_or_min==-1 else None
    print('\n求y',str_)
    x=torch.tensor(0., requires_grad=True)
    a =torch.tensor([3.,-1.] )
    r = torch.tensor([-1.,10.])
    y_list=[]
    x_list=[]
    for i in range(1000):
        y =(torch.relu(x+ a))**0.5@r
        y.backward()
        #print('实数relu开根号方法得到我梯度：',x.grad)
        x=(x+max_or_min*x.grad*lr).clone().detach().requires_grad_(True)
        y_list+=[y.item()]
        x_list+=[x.item()]
        #print('y',y)
        #print('x',x)
    ###
    x=torch.tensor(0., requires_grad=True, dtype=torch.cfloat)
    a =torch.tensor([3.,-1.] , dtype=torch.cfloat)
    r = torch.tensor([-1.,10.], dtype=torch.cfloat)
    y_list_c=[]
    x_list_c=[]
    for i in range(1000):
        y =(x+ a)**0.5@r
        y.real.backward()
        #print('虚数方法得到的梯度：',x.grad)
        x=(x+max_or_min*x.grad*lr).clone().detach().requires_grad_(True)
        y_list_c+=[y.real.item()]
        x_list_c+=[x.real.item()]
        #print('y',y.real.item())
        #print('x',x)
    ####比较二者结果
    fc.plot([y_list,y_list_c],plot_name='求'+str_+'：y过程值比较，lr='+str(lr))
    fc.plot([x_list,x_list_c],plot_name='求'+str_+'：x过程值比较，lr='+str(lr))
    print('实数方法'+str_+'为',max(y_list) if max_or_min==1 else min(y_list) if max_or_min==-1 else None)
    print('复数方法'+str_+'为',max(y_list_c) if max_or_min==1 else min(y_list_c) if max_or_min==-1  else None)
    return
####分析
if __name__=='__main__':
    #先梯度下降求最大值，再求最小值，以及调整学习率的尝试。
    ansys(max_or_min=1,lr=0.02)
    ansys(max_or_min=-1,lr=0.02)
    ansys(max_or_min=-1,lr=0.05)
    ansys(max_or_min=-1,lr=0.005)
####
'''
小结：
★对于求y的最大值
1，复数方法明显占优势，能在全局收敛到最大值，这是因为复数求导机制，使得根号下的负值可以通过虚数梯度获得虚数增长，
从而更快的表现出自身根号外的斜率特征。
2，实数方法由于relu函数的限制，使其无法搜索到远处的最大值的可能。最终收敛到局部极大值。

★对于求y的最小值
1，lr小于0.02时，可以收敛，但当x=0时会因为导数激增，使x值突然变小，不能围绕0点稳定收敛。
2，在lr较大时，由于x回弹太快，当x<-3时，导致整个y值达梯度消失。
3，复变方法不存在梯度不稳定的问题，但是复变方法梯度收敛不到最小值，只能改变为较小的学习率，
但也不如实数方法能更有效的逼近最小值，这是由于复数求导机制中产生的虚部的干扰造成的。

问题：
是否可以利用复分析相关方法，优化梯度下降的收敛稳定性？

'''
####
'''
#fc.plot函数
def plot(data,log=0,label=["y"],k=0,plot_name='未命名'):
    """
    绘制折线图的函数。

    参数:
    data (list): 要绘制的数据列表。
    """
    # 检查输入是否为列表
    #if not isinstance(data, list):
        #raise ValueError("输入必须是列表类型")
    # 绘制折线图
    #time.sleep(0.1)
    if k==1:#转置处理
        data=np.array(data).T
    plt.figure(figsize=(10, 5))  # 设置图形大小
    #if len(np.array(data).shape)==1:
    if isinstance(data[0], Iterable)==0:#不可迭代
        if log==1:
            data=np.log(data)
        plt.plot(data, marker='')  # 绘制折线图，并在每个数据点上标记一个圆点
    else:
        for i in range(len(data)):
            data_i=data[i] if log==0 else np.log(data[i])
            plt.plot(data_i, label=label[i] if len(label)!=1 else None)  # 绘制折线图，并在每个数据点上标记一个圆点
    #plt.plot(data, marker='')  # 绘制折线图，并在每个数据点上标记一个圆点
    plt.legend() if len(label)!=1 else None
    plt.title(plot_name)  # 设置图形标题
    plt.xlabel('Index')  # 设置x轴标签
    plt.ylabel('Value')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形
    time.sleep(0.05)
'''
