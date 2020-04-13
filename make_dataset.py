import scipy
import glob
import numpy as np
import json
from multiprocessing import Pool

def process(args):
    label=args["label"]
    name=args["name"]
    feature=args["feature"]
    filename="npy/"+name+"."+feature+".npy"
    print("[LOAD]",filename)
    f=np.load(filename)
    return name,label,f

def main():
    data=[]
    filename="BirdVox-DCASE-20k.csv"
    feature="spec"
    fp=open(filename)
    next(fp)
    for line in fp:
        arr=line.split(",")
        name=arr[0]
        label=arr[2]
        data.append({"label":label,"name":name,"feature":feature})
        #filename="wav/28d7bccb-30ce-40ae-a33a-c0bb3cb4c1e5.wav"
        #f=get_feature(filename)

    p = Pool(128)
    results=p.map(process, data)
    p.close()

    np.random.seed(1234)

    ml=max([r[2].shape[1]  for r in results])
    feature_num=results[0][2].shape[0]
    n=len(results)

    data=np.zeros((n,feature_num,ml),dtype=np.float32)
    step_data=np.zeros((n,),dtype=np.int32)
    label_data=np.zeros((n,ml),dtype=np.int32)
    name_list=[]
    for i,r in enumerate(results):
        name,label,f=r
        s=f.shape[1]
        data[i,:,:s]=f
        step_data[i]=s
        label_data[i,:]=label
        name_list.append(name)
    print(data.shape)
    data=np.transpose(data,[0,2,1])
    ##
    all_idx=list(range(n))
    np.random.shuffle(all_idx)
    train_idx=all_idx[:n-1000]
    test_idx=all_idx[n-1000:]
    info={}
    info["pid_list_train"]=[name_list[i] for i in train_idx]
    info["pid_list_test"]=[name_list[i] for i in test_idx]
    ##
    train_data=data[train_idx,:,:]
    train_step_data=step_data[train_idx]
    train_label_data=label_data[train_idx,:]
    filename="train_data."+feature+".npy"
    np.save(filename,train_data)
    filename="train_step."+feature+".npy"
    np.save(filename,train_step_data)
    filename="train_label."+feature+".npy"
    np.save(filename,train_label_data)
    ##
    test_data=data[test_idx,:,:]
    test_step_data=step_data[test_idx]
    test_label_data=label_data[test_idx,:]
    filename="test_data."+feature+".npy"
    np.save(filename,test_data)
    filename="test_step."+feature+".npy"
    np.save(filename,test_step_data)
    filename="test_label."+feature+".npy"
    np.save(filename,test_label_data)

    #print(ml)
    fp = open("info."+feature+".json", "w")
    json.dump(info, fp)

if __name__ == '__main__':
    main()
