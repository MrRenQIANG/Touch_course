'''
子类永远在父类前面
如果有多个父类，会根据它们在列表中的顺序被检查
如果对下一个类存在两个合法的选择，选择第一个父类
'''
class Base(object):
    def __init__(self):
        print("enter Base")
        print("leave Base")
class A(Base):
    def __init__(self):
        print("enter A")
        super(A, self).__init__()      # 当你使用 super(cls, inst) 时，Python 会在 inst 的 MRO 列表上搜索 cls 的下一个类。
        print("leave A")
class B(Base):
    def __init__(self):
        print("enter B")
        super(B, self).__init__()
        print("leave B")
class C(A, B):
    def __init__(self):
        print("enter C")
        super(C, self).__init__()
        print ("leave C")

c = C()