import os
from zhouhai.calcute_graph import cal_graph
def Prepare(graphname):
    #dirs=['AE_pretrain','graph','plot','results_train']

    #for dir in dirs:
     #   if not os.path.exists(dir):
      #      os.mkdir(dir)
    if graphname=="handwritten":
        views=[0,1,2,3,4]
            
    if graphname=="mnist":
        views=[0,1,2]
    for view in views:
        if not os.path.exists("graph/%s-%s.txt"%(graphname,view)):
            cal_graph('ncos',view,graphname)


if __name__=="__main__":
    Prepare()