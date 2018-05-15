# coding: utf-8
class foo:
    def __init__(self, color = 'red'):
        self.mycolor = color
        xcolor = color
        
f1 = foo()
f2 = foo(color = 'green')
l = range(23)
l
type(l)
l = list(range(23))
l
f1.mycolor
f2.mycolor
f1.xcolor
class foo2:
    class_color = 'black'
    self.color= 'red'
    
class foo2:
    class_color = 'black'
    def __init__(self,a):
        self.a = a
    def print_a(self):
        print(self.a)
    def print_class_color(self):    		
        print(class_color)
        
f3 = foo2()
f3 = foo2('dghj')
f3.a
f3.print_a()
f3.print_class_color()
f3.class_color
f3.class_color = 'adsf'
f4 = foo2('string2')
f4.a
f3.a
f3.class_color
f4.class_color
f4.class_color = 'dfg'
f3.class_color
foo2.class_color
foo2.a
import pyplot as ppl
import numpy
from matplotlib import pyplot as ppl
fig = ppl.figure()
fig.add_subplot()
fig.show()
fig.axes
gca()
ppl.gca()
ax = ppl.gca()
ax.hist(numpy.random.randn(50000))
show()
ax.draw()
ppl.ion()
