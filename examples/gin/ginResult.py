import numpy as np
# only to get the result of MUTAG dataset
# you should run gin_trainer.py and change the parameter fold_idx from 0 to 9.
# Then run this file to get the 10-fold validation result. 
def main():
    # dataset = 'MUTAG'
    with open('GINMUTAG.txt', 'r') as f:
        targetData = f.readlines()

    idx = 0
    count = 0
    total = {}
    for sigleData in targetData:
        sigleData = sigleData.strip('\n')
        sigleData = sigleData.split(' ')
        if idx == 0:
            total[sigleData[-1]] = float(sigleData[-2])
        else:
            total[sigleData[-1]] += float(sigleData[-2])

        count += 1
        if count == 350:
            idx += 1
            count = 0

    final_epoch = 0
    final_total_acc = 0
    # epoch = 350
    for i in range(350):
        if final_total_acc <= total[str(i)]:
            final_total_acc = total[str(i)]
            final_epoch = i

    index = 0
    arr = np.array([])

    for singleData in targetData:
        singleData = singleData.strip('\n')
        singleData = singleData.split(' ')
        if singleData[-1] == str(final_epoch):
            arr = np.append(arr=arr, values=float(singleData[-2]))
            index += 1

    final_acc = final_total_acc / 10
    std = np.std(a=arr)
    print(final_acc)
    print(std)
    with open('result.txt', 'w') as f:
        f.write('acc=' + str(final_acc) + '   std=' + str(std))


if __name__ == '__main__':
    main()











