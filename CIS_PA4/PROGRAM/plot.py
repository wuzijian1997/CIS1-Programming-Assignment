from matplotlib import pyplot as plt

y1 = [0.01, 0.25, 0.18, 0.43, 0.30, 0.34, 0.37,0.31, 0.68,2.32]
y2 = [0.04, 0.06, 0.01, 0.16, 0.26, 0.25, 0.20,0.19, 0.66, 0.89]
x = [0,10,20,30,40,50,60,70,80,90]
plt.plot(x, y1, color = 'b',lw=2, label = 'ICP w/o RobustPoseEstimation')
plt.plot(x, y2, color = 'green',lw=2,label='ICP w/ RobustPoseEstimation')
plt.ylabel('S Points Error ')
plt.xlabel(' Noisy Ratio (%)')
plt.legend()
plt.savefig('noisy_ratio.png')
plt.show()