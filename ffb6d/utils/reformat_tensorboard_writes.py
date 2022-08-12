from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

#event_acc = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/train_acc/acc_rgbd')
#event_loss_all = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_all')
#event_loss_ctr = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_ctr_of')
#event_loss_kp = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_kp_of')
#event_loss_rgbd = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_rgbd_seg')
#event_loss_target = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_target')

event_acc = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/train_acc/acc_rgbd')
event_loss_all = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_all')
event_loss_ctr = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_ctr_of')
event_loss_kp = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_kp_of')
event_loss_rgbd = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_rgbd_seg')
event_loss_target = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/loss/loss_target')
#event_val_acc = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/val_acc/acc_rgbd')
event_val_acc_avg = EventAccumulator('/home/nachiket/Documents/GitHub/FFB6D/ffb6d/train_log/linemod/train_info/driller/val_acc_avg/acc_rgbd')

event_acc.Reload()
event_loss_all.Reload()
event_loss_ctr.Reload()
event_loss_kp.Reload()
event_loss_rgbd.Reload()
event_loss_target.Reload()
#event_val_acc.Reload()
event_val_acc_avg.Reload()
# Show all tags in the log file
print(event_acc.Tags())
print(event_loss_all.Tags())
#print(event_val_acc.Tags())
print(event_val_acc_avg.Tags())
#print(event_loss_ctr.Tags())
#print(event_loss_kp.Tags())
#print(event_loss_rgbd.Tags())
#print(event_loss_target.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'

def unzip_vals(this_accum, this_tag):
    w_times, step_nums, vals = zip(*this_accum.Scalars(this_tag))
    return w_times, step_nums, vals

w_times, step_nums, vals = unzip_vals(event_acc, 'train_acc')
w1, s1, v1 = unzip_vals(event_loss_rgbd, 'loss')
wt, st, vt = unzip_vals(event_loss_target, 'loss')
wa, sa, va = unzip_vals(event_loss_all, 'loss')
#wv, sv, vv = unzip_vals(event_val_acc, 'val_acc')
wv2, sv2, vv2 = unzip_vals(event_val_acc_avg, 'val_acc_avg')

#print(len(wv), len(sv), len(vv))
print(len(wv2), len(sv2), len(vv2))

print(len(w_times), len(step_nums), len(vals))
print(v1[-20:])

df = pd.DataFrame()
import matplotlib.pyplot as plt

# plot lines
plt.plot(step_nums[-380:], vals[-380:], label="train_acc")
plt.plot(step_nums[-380:], v1[-380:], label="rgbd_loss")
#plt.plot(step_nums[-380:], vt[-380:], label="target_loss")
#plt.plot(step_nums[-380:], va[-380:], label="total_loss")
plt.legend()
plt.show()


#plt.plot(sv, vv, label='val_acc')
plt.plot(sv2, vv2, label='val_acc_avg')
plt.legend()
plt.show()