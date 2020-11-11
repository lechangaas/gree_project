from collections import Counter
from collections import OrderedDict
import matplotlib.pyplot as plt

file_name = "/home/dls1/Desktop/daily_used/20200812/1/label.txt"

label_file = open(file_name, 'r')
label_lines = label_file.readlines()
label_file.close()
label_list = []
for i in label_lines:
    i_strip = i.strip()
    num = i_strip.split(' ')[-1]
    label_list.append(num)

label_count = Counter(label_list)
label_count_sort = sorted(label_count)

label_count_order = OrderedDict()
for i in label_count_sort:
    label_count_order[i] = label_count[i]
plt.figure(figsize=(20, 8), dpi=80)

x_list = list(label_count_order.keys())
y_list = list(label_count_order.values())
plt.bar(x_list, y_list, width=0.5)

for x, y in zip(x_list, y_list):
    plt.text(x, y + 0.05, '%.0f' % y, ha='center', va='bottom', fontsize=15)
plt.xlabel('holes')
plt.ylabel('number')
plt.savefig(fname='label_statistic.jpg', format='jpg')
plt.show()

