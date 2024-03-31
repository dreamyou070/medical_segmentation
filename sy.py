import numpy as np
confusion_matrix = [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]
actual_axis = len(confusion_matrix)
IOU_dict = {}
eps = 1e-15
for actual_idx in range(actual_axis):
    # [1]
    total_actual_num = sum(confusion_matrix[actual_idx])
    # [2] total predicted
    total_predict_num = np.array(confusion_matrix)[:,actual_idx].sum()
    TP = confusion_matrix[actual_idx][actual_idx]
    dice_coeff = 2 * TP  / (total_actual_num + total_predict_num + eps)
    IOU_dict[actual_idx] = round(dice_coeff.item(), 3)
print(IOU_dict)