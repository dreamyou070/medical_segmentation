import numpy as np
confusion_matrix = np.array([[148501218, 12432, 417966, 33333],
[58165, 558565, 65221, 144774],
[694283, 62328, 3115202, 55516],
[72120, 71220, 52321, 848600],])


real_axis, pred_axis = confusion_matrix.shape
values = []
total_nums = []
for r_idx in range(real_axis) :
    predict_result = confusion_matrix[r_idx]
    total_num = predict_result.sum()
    total_nums.append(total_num)
    normalized_value = [round(value/total_num,3) for value in predict_result]
    values.append(normalized_value)
values = np.array(values)
print(values)
print(f'\n classwise total sample = {total_nums}')
