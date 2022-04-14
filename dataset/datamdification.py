
with open('./cityscapes_list/val.txt', 'r') as f:
    data = f.readlines()
    for x in data:
        str1 = '_foggy_beta_0.02'
        str2 = 'leftImg8bit'
        y = x.find(str2)
        print(x)
        z = list(x)
        z.insert(y + 11, str1)
        data1 = ''.join(z)
        with open('./cityscapes_list/val1.txt', 'a') as fw:
            fw.write(data1)